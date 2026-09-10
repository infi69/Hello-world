import cv2
import numpy as np
import inspect
import zxingcpp
from preprocessing_variants import preprocess_variants,cylindrical_unwarp_and_threshold

# ------------------------------------------------------------
# Helpers used by the aggressive decode loop
# ------------------------------------------------------------
def _norm_enum_key(value):
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _resolve_enum_member(enum_obj, value):
    if value is None or enum_obj is None:
        return None
    if not isinstance(value, str):
        return value

    target = _norm_enum_key(value)
    for name in dir(enum_obj):
        if name.startswith("_"):
            continue
        if _norm_enum_key(name) == target:
            return getattr(enum_obj, name)
    return None


def _decode_with_config(image, override=None, base_config=None):
    cfg = {} if base_config is None else dict(base_config)
    if override:
        cfg.update(override)

    kwargs = {}
    if cfg.get("formats") is not None:
        kwargs["formats"] = cfg["formats"]

    kwargs["try_rotate"] = bool(cfg.get("try_rotate", True))
    kwargs["try_downscale"] = bool(cfg.get("try_downscale", True))
    kwargs["try_harder"] = bool(cfg.get("try_harder", True))

    b_enum = getattr(zxingcpp, "Binarizer", None)
    b = _resolve_enum_member(b_enum, cfg.get("binarizer"))
    if b is not None:
        kwargs["binarizer"] = b

    t_enum = getattr(zxingcpp, "TextMode", None)
    t = _resolve_enum_member(t_enum, cfg.get("text_mode"))
    if t is not None:
        kwargs["text_mode"] = t

    try:
        allowed = inspect.signature(zxingcpp.read_barcodes).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    except (TypeError, ValueError):
        pass

    try:
        return zxingcpp.read_barcodes(image, **kwargs)
    except TypeError:
        return zxingcpp.read_barcodes(image)


def _order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _rectify_perspective_from_edges(
    crop_bgr,
    blur_ksize=5,
    canny_low=50,
    canny_high=150,
    min_area_ratio=0.08,
    approx_eps=0.03
):
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0) if blur_ksize > 0 else gray
    edges = cv2.Canny(blur, canny_low, canny_high)

    contours_info = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]
    if not contours:
        return None

    h, w = gray.shape[:2]
    min_area = min_area_ratio * h * w
    best_quad = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, approx_eps * peri, True)

        if len(approx) == 4 and area > best_area:
            best_quad = approx.reshape(4, 2)
            best_area = area

    if best_quad is None:
        return None

    src = _order_quad_points(best_quad)

    wA = np.linalg.norm(src[2] - src[3])
    wB = np.linalg.norm(src[1] - src[0])
    hA = np.linalg.norm(src[1] - src[2])
    hB = np.linalg.norm(src[0] - src[3])

    maxW = max(32, int(round(max(wA, wB))))
    maxH = max(32, int(round(max(hA, hB))))

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32
    )

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        crop_bgr,
        M,
        (maxW, maxH),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return warped


def _rotate_image_keep_size(bgr, angle_deg):
    if angle_deg == 0:
        return bgr

    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        bgr,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )


def _make_decode_views(variant_bgr, pad_fraction, square_size, rotations, min_decode_side=160):
    views = {}
    h0, w0 = variant_bgr.shape[:2]

    if pad_fraction <= 0:
        base = variant_bgr
        base_name = "tight"
    else:
        px = max(2, int(w0 * pad_fraction))
        py = max(2, int(h0 * pad_fraction))
        base = cv2.copyMakeBorder(
            variant_bgr, py, py, px, px,
            borderType=cv2.BORDER_REPLICATE
        )
        base_name = f"pad_{int(round(pad_fraction * 100))}"

    bh, bw = base.shape[:2]
    if min(bh, bw) < min_decode_side:
        scale = float(min_decode_side) / float(max(1, min(bh, bw)))
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        base_name += "_norm"

    views[f"{base_name}|native"] = base
    sq = cv2.resize(base, (square_size, square_size), interpolation=cv2.INTER_CUBIC)
    views[f"{base_name}|sq{square_size}"] = sq

    rotated_views = {}
    for name, v in views.items():
        for ang in rotations:
            rv = _rotate_image_keep_size(v, ang)
            rotated_views[f"{name}|r{ang:+d}"] = rv

    return rotated_views


def _sharpen_image(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def _barcode_likeness_score(img):
    """
    Heuristic score: higher means more likely to be a barcode.
    This is used to stop when the ROI has no barcode memory / quiet zone.
    """
    if img is None or img.size == 0:
        return 0.0

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return 0.0

    # Contrast
    contrast = float(gray.std())

    # Spatial frequency / edges
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())

    # Binary mask and quiet-zone clues
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = max(3, int(0.12 * min(h, w)))

    quiet_zone_score = 1.0 - np.mean([
        (bw[:, :border] > 0).mean(),
        (bw[:, -border:] > 0).mean(),
        (bw[:border, :] > 0).mean(),
        (bw[-border:, :] > 0).mean(),
    ])

    bright = gray > 220
    glare_border_ratio = np.mean([
        bright[:, :border].mean(),
        bright[:, -border:].mean(),
        bright[:border, :].mean(),
        bright[-border:, :].mean(),
    ])

    edges = cv2.Canny(gray, 50, 200)
    edge_density = float((edges > 0).mean())

    # score in 0..1
    score = (
        0.35 * quiet_zone_score +
        0.20 * min(1.0, contrast / 35.0) +
        0.20 * min(1.0, lap_var / 200.0) +
        0.15 * max(0.0, 1.0 - glare_border_ratio) +
        0.10 * max(0.0, 1.0 - abs(edge_density - 0.18) / 0.25)
    )

    return float(score)


# ------------------------------------------------------------
# Main aggressive decoder
# ------------------------------------------------------------
def decode_roi_until_success(
    roi,
    max_passes=30,
    debug=False,
    decode_threshold=0.35,
    stop_when_score_below=0.12,
    max_no_score_passes=4,
):
    """
    Keep adjusting the ROI until:
      1) ZXing decodes successfully, OR
      2) the ROI no longer looks barcode-like enough
    """
    if roi is None or roi.size == 0:
        return {
            "decoded": False,
            "result": None,
            "reason": "empty_roi",
            "score": 0.0,
            "attempts": 0,
        }

    base_cfg = {
        "formats": None,
        "try_rotate": True,
        "try_downscale": True,
        "try_harder": True,
        "binarizer": "LOCAL_AVERAGE",
        "text_mode": None,
    }

    # Search ranges for aggressive but controlled tuning
    blur_ksizes = [0, 1, 3, 5, 7, 9]
    canny_lows = [20, 30, 40, 50, 60]
    canny_highs = [80, 100, 120, 150, 180]
    area_ratios = [0.03, 0.05, 0.08, 0.12, 0.16]
    approx_eps = [0.02, 0.03, 0.04, 0.05]

    pad_fractions = [0.0, 0.05, 0.10, 0.18, 0.25]
    square_sizes = [160, 192, 256, 320]
    rotations = [0, -6, 6, -12, 12, -18, 18, -30, 30]

    fallback_overrides = [
        {},
        {"binarizer": "GLOBAL_HISTOGRAM", "try_downscale": False},
        {"try_rotate": False, "try_downscale": False},
        {"binarizer": "LOCAL_AVERAGE", "try_downscale": True},
    ]

    best_text = None
    best_format = None
    best_meta = None
    best_score = -1.0

    no_score_passes = 0

    for pass_idx in range(1, max_passes + 1):
        if debug:
            print(f"\n=== PASS {pass_idx}/{max_passes} ===")

        # Geometry tuning
        blur_ksize = blur_ksizes[(pass_idx - 1) % len(blur_ksizes)]
        canny_low = canny_lows[(pass_idx - 1) % len(canny_lows)]
        canny_high = canny_highs[(pass_idx - 1) % len(canny_highs)]
        min_area_ratio = area_ratios[(pass_idx - 1) % len(area_ratios)]
        approx_e = approx_eps[(pass_idx - 1) % len(approx_eps)]

        pad_fraction = pad_fractions[(pass_idx - 1) % len(pad_fractions)]
        square_size = square_sizes[(pass_idx - 1) % len(square_sizes)]
        rotation = rotations[(pass_idx - 1) % len(rotations)]

        # Build a batch of candidate crops for this pass
        candidate_crops = {"raw": roi}

        # 1) perspective rectification using tuned contour params
        rectified = _rectify_perspective_from_edges(
            roi,
            blur_ksize=blur_ksize,
            canny_low=canny_low,
            canny_high=canny_high,
            min_area_ratio=min_area_ratio,
            approx_eps=approx_e,
        )
        if rectified is not None and rectified.size > 0:
            candidate_crops["rectified"] = rectified

        # 2) if raw crop is too weak, also try a few light manipulations
        if pass_idx % 2 == 0:
            candidate_crops["rotated"] = _rotate_image_keep_size(roi, rotation)
            candidate_crops["sharpened"] = _sharpen_image(roi)

        # 3) always use your existing preprocessing variants on raw and rectified
        for crop_name, crop in list(candidate_crops.items()):
            variants = preprocess_variants(crop)
            for vname, vimg in variants.items():
                candidate_crops[f"{crop_name}|{vname}"] = vimg

        # Try all candidates in this pass
        pass_best_result = None
        pass_best_score = -1.0

        for crop_name, crop_img in candidate_crops.items():
            if crop_img is None or crop_img.size == 0:
                continue

            score = _barcode_likeness_score(crop_img)
            if debug:
                print(f"  crop={crop_name:20s} barcode_score={score:.3f}")

            # If crop is too weak to be a barcode, skip extra work
            if score < stop_when_score_below and pass_idx >= 2:
                continue

            variant_images = preprocess_variants(crop_img)
            for variant_name, variant in variant_images.items():
                if variant is None or variant.size == 0:
                    continue

                decode_views = _make_decode_views(
                    variant,
                    pad_fraction=pad_fraction,
                    square_size=square_size,
                    rotations=(0, rotation, -rotation) if rotation != 0 else (0,),
                )

                for view_name, view in decode_views.items():
                    for override_idx, override in enumerate(fallback_overrides):
                        decoded = _decode_with_config(view, override=override, base_config=base_cfg)

                        if not decoded:
                            continue

                        for d in decoded:
                            txt = (d.text or "").strip()
                            if not txt:
                                continue

                            if debug:
                                print(f"    SUCCESS: view={view_name}, override#{override_idx} -> {txt}")

                            candidate_score = _barcode_likeness_score(view)
                            if candidate_score > pass_best_score:
                                pass_best_score = candidate_score
                                pass_best_result = {
                                    "text": txt,
                                    "format": str(d.format),
                                    "source": "adaptive_decoder",
                                    "variant": f"{crop_name}|{variant_name}|{view_name}|{override}",
                                    "score": candidate_score,
                                }

                            if txt and (best_text is None or candidate_score > best_score):
                                best_text = txt
                                best_format = str(d.format)
                                best_meta = {
                                    "variant": f"{crop_name}|{variant_name}|{view_name}|{override}",
                                    "score": candidate_score,
                                }
                                best_score = candidate_score

                            # Early break on successful decode
                            return {
                                "decoded": True,
                                "result": {
                                    "text": txt,
                                    "format": str(d.format),
                                    "source": "adaptive_decoder",
                                    "variant": f"{crop_name}|{variant_name}|{view_name}|{override}",
                                    "score": candidate_score,
                                },
                                "reason": "barcode_decoded",
                                "score": candidate_score,
                                "attempts": pass_idx,
                            }

        # If the whole pass produced zero decent barcode-like candidates,
        # stop early instead of continuing blindly
        if pass_best_result is None:
            no_score_passes += 1
        else:
            no_score_passes = 0

        if no_score_passes >= max_no_score_passes:
            if debug:
                print("Stopping: ROI no longer looks barcode-like enough.")
            break

        # Also stop if the ROI is too weak overall
        worst_case_score = min(
            _barcode_likeness_score(c) for c in candidate_crops.values() if c is not None and c.size > 0
        ) if candidate_crops else 0.0

        if worst_case_score < decode_threshold and pass_idx >= 3:
            if debug:
                print(f"Stopping: overall barcode-likeness score too low ({worst_case_score:.3f})")
            break

    if best_text is not None:
        return {
            "decoded": True,
            "result": {
                "text": best_text,
                "format": best_format,
                "source": "adaptive_decoder",
                "variant": best_meta["variant"],
                "score": best_score,
            },
            "reason": "best_decode_found",
            "score": best_score,
            "attempts": max_passes,
        }

    return {
        "decoded": False,
        "result": None,
        "reason": "no_barcode_like_roi_or_no_decode",
        "score": float(min(_barcode_likeness_score(v) for v in [roi] if v is not None and v.size > 0)) if roi is not None and roi.size > 0 else 0.0,
        "attempts": max_passes,
    }
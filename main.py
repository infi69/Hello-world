"""
Self-contained DataMatrix decode pipeline for pre-cropped ROIs in test-crops/.

Strategy:
  1) Generate preprocessing variants and try ZXing (DataMatrix only) — fast path
  2) On failure, run an aggressive multi-pass decoder (perspective / pad / rotate / configs)

Outputs:
  - Console progress
  - CSV report with timings and winning method
  - Detailed step/variant timing log
"""

from __future__ import annotations

import argparse
import csv
import inspect
import logging
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import zxingcpp

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CROPS_DIR = SCRIPT_DIR / "test-crops"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Prefer DataMatrix-only when the installed zxing-cpp exposes format enums.
_DM_FORMAT = None
for _attr in ("BarcodeFormat", "Format"):
    _enum = getattr(zxingcpp, _attr, None)
    if _enum is None:
        continue
    for _name in ("DataMatrix", "DATA_MATRIX", "Datamatrix"):
        if hasattr(_enum, _name):
            _DM_FORMAT = getattr(_enum, _name)
            break
    if _DM_FORMAT is not None:
        break


# ===========================================================================
# Preprocessing variants (self-contained; mirrored from developer experiment)
# ===========================================================================
def cylindrical_unwarp_and_threshold(roi: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = xx - cx
    dy = yy - cy
    theta = np.arctan2(dy, dx + 1e-6)

    unwrapped_x = ((theta + np.pi) / (2 * np.pi)) * (w - 1)
    unwrapped_y = yy

    unwrapped = cv2.remap(
        gray,
        unwrapped_x,
        unwrapped_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7)).apply(unwrapped)
    _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def preprocess_variants(img: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    variants = {
        "original": img,
        "grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "hist_eq": cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR),
        "gaussian_blur": cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR),
        "inverted": cv2.cvtColor(cv2.bitwise_not(gray), cv2.COLOR_GRAY2BGR),
        "sharpened": cv2.cvtColor(cv2.filter2D(gray, -1, kernel_sharpen), cv2.COLOR_GRAY2BGR),
        "adaptive_thresh": cv2.cvtColor(
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8
            ),
            cv2.COLOR_GRAY2BGR,
        ),
        "clahe": cv2.cvtColor(
            cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(gray),
            cv2.COLOR_GRAY2BGR,
        ),
        "hsv_value": cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR),
        "hsv_saturation": cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR),
        "lab_luminance": cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR),
        "cylindrical_unwrap": cylindrical_unwarp_and_threshold(img),
    }

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)
    th = cv2.adaptiveThreshold(
        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 2
    )
    variants["clahe_adaptive_thresh"] = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return variants


# ===========================================================================
# ZXing helpers (DataMatrix only)
# ===========================================================================
def _norm_enum_key(value: Any) -> str:
    return "".join(ch for ch in str(value).upper() if ch.isalnum())


def _resolve_enum_member(enum_obj: Any, value: Any) -> Any:
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


def _is_datamatrix(fmt: Any) -> bool:
    key = _norm_enum_key(fmt)
    return "DATAMATRIX" in key


def _decode_with_config(image: np.ndarray, override: dict | None = None, base_config: dict | None = None):
    cfg = {} if base_config is None else dict(base_config)
    if override:
        cfg.update(override)

    kwargs: dict[str, Any] = {}
    formats = cfg.get("formats")
    if formats is not None:
        kwargs["formats"] = formats
    elif _DM_FORMAT is not None:
        kwargs["formats"] = _DM_FORMAT

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
        results = zxingcpp.read_barcodes(image, **kwargs)
    except TypeError:
        results = zxingcpp.read_barcodes(image)

    # Always filter to DataMatrix even if format kwarg unsupported.
    filtered = []
    for d in results or []:
        if _is_datamatrix(getattr(d, "format", "")):
            filtered.append(d)
        elif _DM_FORMAT is None and not getattr(d, "format", None):
            # Extremely old bindings may lack format; keep non-empty text cautiously.
            txt = (getattr(d, "text", None) or "").strip()
            if txt:
                filtered.append(d)
    return filtered


def _order_quad_points(pts: np.ndarray) -> np.ndarray:
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _rectify_perspective_from_edges(
    crop_bgr: np.ndarray,
    blur_ksize: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
    min_area_ratio: float = 0.08,
    approx_eps: float = 0.03,
) -> np.ndarray | None:
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
    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        crop_bgr, M, (maxW, maxH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _rotate_image_keep_size(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if angle_deg == 0:
        return bgr
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _make_decode_views(
    variant_bgr: np.ndarray,
    pad_fraction: float,
    square_size: int,
    rotations: tuple[int, ...] | list[int],
    min_decode_side: int = 160,
) -> dict[str, np.ndarray]:
    views: dict[str, np.ndarray] = {}
    h0, w0 = variant_bgr.shape[:2]

    if pad_fraction <= 0:
        base = variant_bgr
        base_name = "tight"
    else:
        px = max(2, int(w0 * pad_fraction))
        py = max(2, int(h0 * pad_fraction))
        base = cv2.copyMakeBorder(variant_bgr, py, py, px, px, borderType=cv2.BORDER_REPLICATE)
        base_name = f"pad_{int(round(pad_fraction * 100))}"

    bh, bw = base.shape[:2]
    if min(bh, bw) < min_decode_side:
        scale = float(min_decode_side) / float(max(1, min(bh, bw)))
        base = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        base_name += "_norm"

    views[f"{base_name}|native"] = base
    views[f"{base_name}|sq{square_size}"] = cv2.resize(
        base, (square_size, square_size), interpolation=cv2.INTER_CUBIC
    )

    rotated_views: dict[str, np.ndarray] = {}
    for name, v in views.items():
        for ang in rotations:
            rotated_views[f"{name}|r{ang:+d}"] = _rotate_image_keep_size(v, ang)
    return rotated_views


def _sharpen_image(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(img, -1, kernel)


def _barcode_likeness_score(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return 0.0

    contrast = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = max(3, int(0.12 * min(h, w)))
    quiet_zone_score = 1.0 - np.mean(
        [
            (bw[:, :border] > 0).mean(),
            (bw[:, -border:] > 0).mean(),
            (bw[:border, :] > 0).mean(),
            (bw[-border:, :] > 0).mean(),
        ]
    )
    bright = gray > 220
    glare_border_ratio = np.mean(
        [
            bright[:, :border].mean(),
            bright[:, -border:].mean(),
            bright[:border, :].mean(),
            bright[-border:, :].mean(),
        ]
    )
    edge_density = float((cv2.Canny(gray, 50, 200) > 0).mean())
    score = (
        0.35 * quiet_zone_score
        + 0.20 * min(1.0, contrast / 35.0)
        + 0.20 * min(1.0, lap_var / 200.0)
        + 0.15 * max(0.0, 1.0 - glare_border_ratio)
        + 0.10 * max(0.0, 1.0 - abs(edge_density - 0.18) / 0.25)
    )
    return float(score)


# ===========================================================================
# Decode strategies
# ===========================================================================
@dataclass
class StepLog:
    phase: str
    step: str
    elapsed_ms: float
    success: bool
    detail: str = ""


@dataclass
class DecodeResult:
    filename: str
    decoded: bool
    text: str = ""
    format: str = ""
    method: str = ""
    variant: str = ""
    reason: str = ""
    simple_ms: float = 0.0
    aggressive_ms: float = 0.0
    total_ms: float = 0.0
    variants_tried: int = 0
    aggressive_attempts: int = 0
    error: str = ""
    steps: list[StepLog] = field(default_factory=list)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)


def _image_quality_metrics(img: np.ndarray) -> dict[str, float]:
    """Return barcode-likeness metrics for a crop or decode view."""
    if img is None or img.size == 0:
        return {
            "contrast": 0.0,
            "lap_var": 0.0,
            "quiet_zone_score": 0.0,
            "glare_border_ratio": 0.0,
            "edge_density": 0.0,
            "barcode_score": 0.0,
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    h, w = gray.shape[:2]
    if h < 10 or w < 10:
        return {
            "contrast": 0.0,
            "lap_var": 0.0,
            "quiet_zone_score": 0.0,
            "glare_border_ratio": 0.0,
            "edge_density": 0.0,
            "barcode_score": 0.0,
        }

    contrast = float(gray.std())
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = max(3, int(0.12 * min(h, w)))
    quiet_zone_score = 1.0 - np.mean(
        [
            (bw[:, :border] > 0).mean(),
            (bw[:, -border:] > 0).mean(),
            (bw[:border, :] > 0).mean(),
            (bw[-border:, :] > 0).mean(),
        ]
    )
    bright = gray > 220
    glare_border_ratio = np.mean(
        [
            bright[:, :border].mean(),
            bright[:, -border:].mean(),
            bright[:border, :].mean(),
            bright[-border:, :].mean(),
        ]
    )
    edge_density = float((cv2.Canny(gray, 50, 200) > 0).mean())
    barcode_score = float(
        0.35 * quiet_zone_score
        + 0.20 * min(1.0, contrast / 35.0)
        + 0.20 * min(1.0, lap_var / 200.0)
        + 0.15 * max(0.0, 1.0 - glare_border_ratio)
        + 0.10 * max(0.0, 1.0 - abs(edge_density - 0.18) / 0.25)
    )

    return {
        "contrast": contrast,
        "lap_var": lap_var,
        "quiet_zone_score": float(quiet_zone_score),
        "glare_border_ratio": float(glare_border_ratio),
        "edge_density": edge_density,
        "barcode_score": barcode_score,
    }


def _simple_variant_decode(roi: np.ndarray, steps: list[StepLog]) -> dict[str, Any]:
    """Fast path: preprocess variants -> ZXing DataMatrix."""
    base_cfg = {
        "formats": _DM_FORMAT,
        "try_rotate": True,
        "try_downscale": True,
        "try_harder": True,
        "binarizer": "LOCAL_AVERAGE",
    }
    overrides = [
        {},
        {"binarizer": "GLOBAL_HISTOGRAM"},
        {"binarizer": "LOCAL_AVERAGE", "try_downscale": False},
    ]

    t0 = time.perf_counter()
    variants = preprocess_variants(roi)
    steps.append(
        StepLog(
            phase="simple",
            step="generate_variants",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            success=True,
            detail=f"count={len(variants)} names={','.join(variants.keys())}",
        )
    )

    tried = 0
    for vname, vimg in variants.items():
        for oi, override in enumerate(overrides):
            tried += 1
            t_try = time.perf_counter()
            decoded = _decode_with_config(vimg, override=override, base_config=base_cfg)
            elapsed = (time.perf_counter() - t_try) * 1000.0
            override_label = override.get("binarizer", "default")
            if decoded:
                for d in decoded:
                    txt = (d.text or "").strip()
                    if not txt:
                        continue
                    steps.append(
                        StepLog(
                            phase="simple",
                            step=f"variant:{vname}|cfg:{override_label}",
                            elapsed_ms=elapsed,
                            success=True,
                            detail=f"text={txt[:80]}",
                        )
                    )
                    return {
                        "decoded": True,
                        "text": txt,
                        "format": str(getattr(d, "format", "DataMatrix")),
                        "variant": f"{vname}|{override_label}",
                        "variants_tried": tried,
                        "steps": steps,
                    }
            steps.append(
                StepLog(
                    phase="simple",
                    step=f"variant:{vname}|cfg:{override_label}",
                    elapsed_ms=elapsed,
                    success=False,
                    detail="no_decode",
                )
            )

    return {
        "decoded": False,
        "text": "",
        "format": "",
        "variant": "",
        "variants_tried": tried,
        "steps": steps,
    }


def _aggressive_decode(
    roi: np.ndarray,
    steps: list[StepLog],
    max_passes: int = 12,
    decode_threshold: float = 0.35,
    stop_when_score_below: float = 0.12,
    max_no_score_passes: int = 4,
) -> dict[str, Any]:
    """Fallback: multi-pass geometry + variants + decode views (DataMatrix only)."""
    candidate_rows: list[dict[str, Any]] = []

    base_cfg = {
        "formats": _DM_FORMAT,
        "try_rotate": True,
        "try_downscale": True,
        "try_harder": True,
        "binarizer": "LOCAL_AVERAGE",
        "text_mode": None,
    }

    blur_ksizes = [0, 1, 3, 5, 7, 9]
    canny_lows = [20, 30, 40, 50, 60]
    canny_highs = [80, 100, 120, 150, 180]
    area_ratios = [0.03, 0.05, 0.08, 0.12, 0.16]
    approx_eps_list = [0.02, 0.03, 0.04, 0.05]
    pad_fractions = [0.0, 0.05, 0.10, 0.18, 0.25]
    square_sizes = [160, 192, 256, 320]
    rotations = [0, -6, 6, -12, 12, -18, 18, -30, 30]
    fallback_overrides = [
        {},
        {"binarizer": "GLOBAL_HISTOGRAM", "try_downscale": False},
        {"try_rotate": False, "try_downscale": False},
        {"binarizer": "LOCAL_AVERAGE", "try_downscale": True},
    ]

    no_score_passes = 0

    for pass_idx in range(1, max_passes + 1):
        t_pass = time.perf_counter()
        blur_ksize = blur_ksizes[(pass_idx - 1) % len(blur_ksizes)]
        canny_low = canny_lows[(pass_idx - 1) % len(canny_lows)]
        canny_high = canny_highs[(pass_idx - 1) % len(canny_highs)]
        min_area_ratio = area_ratios[(pass_idx - 1) % len(area_ratios)]
        approx_e = approx_eps_list[(pass_idx - 1) % len(approx_eps_list)]
        pad_fraction = pad_fractions[(pass_idx - 1) % len(pad_fractions)]
        square_size = square_sizes[(pass_idx - 1) % len(square_sizes)]
        rotation = rotations[(pass_idx - 1) % len(rotations)]

        candidate_crops: dict[str, np.ndarray] = {"raw": roi}
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
        if pass_idx % 2 == 0:
            candidate_crops["rotated"] = _rotate_image_keep_size(roi, rotation)
            candidate_crops["sharpened"] = _sharpen_image(roi)

        # Expand each geometry crop with preprocessing variants once
        # (avoid nested variant-of-variant explosion that makes passes too slow).
        variant_crops: dict[str, np.ndarray] = {}
        for crop_name, crop in candidate_crops.items():
            for vname, vimg in preprocess_variants(crop).items():
                variant_crops[f"{crop_name}|{vname}"] = vimg

        pass_best_score = -1.0
        got_any_decode_attempt = False

        for crop_name, crop_img in variant_crops.items():
            if crop_img is None or crop_img.size == 0:
                continue
            score = _barcode_likeness_score(crop_img)
            metrics = _image_quality_metrics(crop_img)
            if score < stop_when_score_below and pass_idx >= 2:
                continue
            pass_best_score = max(pass_best_score, score)

            candidate_rows.append(
                {
                    "image_name": "",
                    "roi_name": "ROI_01",
                    "roi_variant": crop_name,
                    "view_name": f"{crop_name}|native",
                    "decoded": False,
                    "text": "",
                    "format": "",
                    "method": "aggressive_fallback",
                    "reason": "candidate_evaluated",
                    "contrast": metrics["contrast"],
                    "lap_var": metrics["lap_var"],
                    "quiet_zone_score": metrics["quiet_zone_score"],
                    "glare_border_ratio": metrics["glare_border_ratio"],
                    "edge_density": metrics["edge_density"],
                    "barcode_score": metrics["barcode_score"],
                    "pass_idx": pass_idx,
                    "rotation_deg": int(rotation),
                    "roi_w": int(crop_img.shape[1]),
                    "roi_h": int(crop_img.shape[0]),
                }
            )

            decode_views = _make_decode_views(
                crop_img,
                pad_fraction=pad_fraction,
                square_size=square_size,
                rotations=(0, rotation, -rotation) if rotation != 0 else (0,),
            )
            for view_name, view in decode_views.items():
                for override in fallback_overrides:
                    got_any_decode_attempt = True
                    t_try = time.perf_counter()
                    decoded = _decode_with_config(view, override=override, base_config=base_cfg)
                    elapsed = (time.perf_counter() - t_try) * 1000.0
                    label = (
                        f"pass{pass_idx}|{crop_name}|{view_name}|"
                        f"{override.get('binarizer', 'default')}"
                    )
                    view_metrics = _image_quality_metrics(view)
                    if decoded:
                        for d in decoded:
                            txt = (d.text or "").strip()
                            if not txt:
                                continue
                            candidate_score = _barcode_likeness_score(view)
                            steps.append(
                                StepLog(
                                    phase="aggressive",
                                    step=label,
                                    elapsed_ms=elapsed,
                                    success=True,
                                    detail=f"text={txt[:80]} score={candidate_score:.3f}",
                                )
                            )
                            steps.append(
                                StepLog(
                                    phase="aggressive",
                                    step=f"pass_{pass_idx}_complete",
                                    elapsed_ms=(time.perf_counter() - t_pass) * 1000.0,
                                    success=True,
                                    detail="decoded",
                                )
                            )
                            candidate_rows.append(
                                {
                                    "image_name": "",
                                    "roi_name": "ROI_01",
                                    "roi_variant": crop_name,
                                    "view_name": view_name,
                                    "decoded": True,
                                    "text": txt,
                                    "format": str(getattr(d, "format", "DataMatrix")),
                                    "method": "aggressive_fallback",
                                    "reason": "barcode_decoded",
                                    "contrast": view_metrics["contrast"],
                                    "lap_var": view_metrics["lap_var"],
                                    "quiet_zone_score": view_metrics["quiet_zone_score"],
                                    "glare_border_ratio": view_metrics["glare_border_ratio"],
                                    "edge_density": view_metrics["edge_density"],
                                    "barcode_score": view_metrics["barcode_score"],
                                    "pass_idx": pass_idx,
                                    "rotation_deg": int(rotation),
                                    "roi_w": int(view.shape[1]),
                                    "roi_h": int(view.shape[0]),
                                }
                            )
                            return {
                                "decoded": True,
                                "text": txt,
                                "format": str(getattr(d, "format", "DataMatrix")),
                                "variant": label,
                                "attempts": pass_idx,
                                "steps": steps,
                                "reason": "barcode_decoded",
                                "candidate_rows": candidate_rows,
                            }
                    # Sparse failure logs: native/default only
                    if view_name.endswith("|native|r+0") and not override:
                        steps.append(
                            StepLog(
                                phase="aggressive",
                                step=label,
                                elapsed_ms=elapsed,
                                success=False,
                                detail=f"score={score:.3f}",
                            )
                        )

        pass_elapsed = (time.perf_counter() - t_pass) * 1000.0
        steps.append(
            StepLog(
                phase="aggressive",
                step=f"pass_{pass_idx}_complete",
                elapsed_ms=pass_elapsed,
                success=False,
                detail=f"crops={len(variant_crops)} best_score={pass_best_score:.3f}",
            )
        )

        if not got_any_decode_attempt or pass_best_score < 0:
            no_score_passes += 1
        else:
            no_score_passes = 0

        if no_score_passes >= max_no_score_passes:
            steps.append(
                StepLog(
                    phase="aggressive",
                    step="early_stop",
                    elapsed_ms=0.0,
                    success=False,
                    detail="roi_not_barcode_like",
                )
            )
            break

        scores = [
            _barcode_likeness_score(c)
            for c in variant_crops.values()
            if c is not None and c.size > 0
        ]
        worst = min(scores) if scores else 0.0
        if worst < decode_threshold and pass_idx >= 3:
            steps.append(
                StepLog(
                    phase="aggressive",
                    step="early_stop",
                    elapsed_ms=0.0,
                    success=False,
                    detail=f"low_score={worst:.3f}",
                )
            )
            break

    return {
        "decoded": False,
        "text": "",
        "format": "",
        "variant": "",
        "attempts": max_passes,
        "steps": steps,
        "reason": "no_barcode_like_roi_or_no_decode",
        "candidate_rows": candidate_rows,
    }


def _make_success_candidate_row(
    image_name: str,
    text: str,
    fmt: str,
    method: str,
    variant: str,
    reason: str,
    image: np.ndarray,
    pass_idx: int = 0,
    rotation_deg: int = 0,
) -> dict[str, Any]:
    metrics = _image_quality_metrics(image)
    return {
        "image_name": image_name,
        "roi_name": "ROI_01",
        "roi_variant": variant or "final",
        "view_name": "final",
        "decoded": True,
        "text": text,
        "format": fmt,
        "method": method,
        "reason": reason,
        "contrast": metrics["contrast"],
        "lap_var": metrics["lap_var"],
        "quiet_zone_score": metrics["quiet_zone_score"],
        "glare_border_ratio": metrics["glare_border_ratio"],
        "edge_density": metrics["edge_density"],
        "barcode_score": metrics["barcode_score"],
        "pass_idx": pass_idx,
        "rotation_deg": rotation_deg,
        "roi_w": int(image.shape[1]),
        "roi_h": int(image.shape[0]),
    }


def decode_one_crop(image_path: str | Path, aggressive_max_passes: int = 12) -> DecodeResult:
    path = Path(image_path)
    result = DecodeResult(filename=path.name, decoded=False)
    t_total = time.perf_counter()
    steps: list[StepLog] = []

    try:
        t_load = time.perf_counter()
        img = cv2.imread(str(path))
        steps.append(
            StepLog(
                phase="io",
                step="imread",
                elapsed_ms=(time.perf_counter() - t_load) * 1000.0,
                success=img is not None,
                detail=f"shape={None if img is None else img.shape}",
            )
        )
        if img is None:
            result.reason = "imread_failed"
            result.steps = steps
            result.total_ms = (time.perf_counter() - t_total) * 1000.0
            return result

        # Phase 1: simple variants
        t_simple = time.perf_counter()
        simple = _simple_variant_decode(img, steps)
        result.simple_ms = (time.perf_counter() - t_simple) * 1000.0
        result.variants_tried = int(simple.get("variants_tried", 0))
        steps = simple["steps"]

        if simple["decoded"]:
            result.decoded = True
            result.text = simple["text"]
            result.format = simple["format"]
            result.method = "simple_variants"
            result.variant = simple["variant"]
            result.reason = "decoded_simple"
            result.candidate_rows = [
                _make_success_candidate_row(
                    image_name=path.name,
                    text=simple["text"],
                    fmt=simple["format"],
                    method="simple_variants",
                    variant=simple["variant"],
                    reason="decoded_simple",
                    image=img,
                )
            ]
            result.steps = steps
            result.total_ms = (time.perf_counter() - t_total) * 1000.0
            return result

        # Phase 2: aggressive fallback
        t_agg = time.perf_counter()
        aggressive = _aggressive_decode(img, steps, max_passes=aggressive_max_passes)
        result.aggressive_ms = (time.perf_counter() - t_agg) * 1000.0
        result.aggressive_attempts = int(aggressive.get("attempts", 0))
        steps = aggressive["steps"]
        result.candidate_rows = aggressive.get("candidate_rows", [])

        for candidate in result.candidate_rows:
            candidate["image_name"] = path.name
            candidate["roi_name"] = candidate.get("roi_name", "ROI_01")

        if aggressive["decoded"]:
            result.decoded = True
            result.text = aggressive["text"]
            result.format = aggressive["format"]
            result.method = "aggressive_fallback"
            result.variant = aggressive["variant"]
            result.reason = aggressive.get("reason", "decoded_aggressive")
            if not any(bool(c.get("decoded")) for c in result.candidate_rows):
                result.candidate_rows.append(
                    _make_success_candidate_row(
                        image_name=path.name,
                        text=aggressive["text"],
                        fmt=aggressive["format"],
                        method="aggressive_fallback",
                        variant=aggressive["variant"],
                        reason=aggressive.get("reason", "decoded_aggressive"),
                        image=img,
                        pass_idx=int(aggressive.get("attempts", 0)),
                    )
                )
        else:
            result.method = "none"
            result.reason = aggressive.get("reason", "failed")

        result.steps = steps
        result.total_ms = (time.perf_counter() - t_total) * 1000.0
        return result

    except Exception as exc:  # noqa: BLE001 — worker must return structured error
        result.error = f"{type(exc).__name__}: {exc}"
        result.reason = "exception"
        result.method = "error"
        steps.append(
            StepLog(
                phase="error",
                step="exception",
                elapsed_ms=0.0,
                success=False,
                detail=traceback.format_exc(limit=5),
            )
        )
        result.steps = steps
        result.total_ms = (time.perf_counter() - t_total) * 1000.0
        return result


def _worker_decode(args: tuple[str, int]) -> dict[str, Any]:
    """ProcessPool entrypoint — returns a picklable dict."""
    image_path, max_passes = args
    res = decode_one_crop(image_path, aggressive_max_passes=max_passes)
    payload = asdict(res)
    return payload


# ===========================================================================
# Batching / multiprocessing / reporting
# ===========================================================================
def list_crop_images(crops_dir: Path) -> list[Path]:
    files = [
        p
        for p in sorted(crops_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    return files


def chunked(items: list[Path], batch_size: int) -> list[list[Path]]:
    if batch_size <= 0:
        return [items]
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("datamatrix_decode")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _row_name(row: dict[str, Any]) -> str:
    return str(row.get("filename") or row.get("image_name") or row.get("roi_name") or "unknown")


def write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    metric_fieldnames = [
        "image_name",
        "roi_name",
        "roi_variant",
        "view_name",
        "decoded",
        "text",
        "format",
        "method",
        "reason",
        "contrast",
        "lap_var",
        "quiet_zone_score",
        "glare_border_ratio",
        "edge_density",
        "barcode_score",
        "pass_idx",
        "rotation_deg",
        "roi_w",
        "roi_h",
    ]
    legacy_fieldnames = [
        "filename",
        "decoded",
        "text",
        "format",
        "method",
        "variant",
        "reason",
        "simple_ms",
        "aggressive_ms",
        "total_ms",
        "variants_tried",
        "aggressive_attempts",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fieldnames + legacy_fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if row.get("image_name") is not None or row.get("roi_name") is not None or row.get("roi_variant") is not None:
                out = {k: row.get(k, "") for k in metric_fieldnames}
                out["decoded"] = bool(out["decoded"])
                for key in ("contrast", "lap_var", "quiet_zone_score", "glare_border_ratio", "edge_density", "barcode_score", "pass_idx", "rotation_deg", "roi_w", "roi_h"):
                    try:
                        if out[key] in (None, ""):
                            out[key] = ""
                        else:
                            out[key] = float(out[key]) if key not in {"pass_idx", "rotation_deg", "roi_w", "roi_h"} else int(float(out[key]))
                    except (TypeError, ValueError):
                        pass
                out.update({k: row.get(k, "") for k in legacy_fieldnames})
                out["decoded"] = bool(out["decoded"])
                writer.writerow(out)
            else:
                out = {k: row.get(k, "") for k in legacy_fieldnames}
                out["decoded"] = bool(out["decoded"])
                for ms_key in ("simple_ms", "aggressive_ms", "total_ms"):
                    try:
                        out[ms_key] = f"{float(out[ms_key]):.2f}"
                    except (TypeError, ValueError):
                        out[ms_key] = out[ms_key]
                writer.writerow(out)


def log_result_steps(logger: logging.Logger, row: dict[str, Any]) -> None:
    fname = row.get("filename") or row.get("image_name") or "unknown"
    logger.info(
        "RESULT %s | decoded=%s method=%s total_ms=%.2f text=%r",
        fname,
        row.get("decoded"),
        row.get("method"),
        float(row.get("total_ms") or 0.0),
        (row.get("text") or "")[:120],
    )
    for step in row.get("steps") or []:
        logger.debug(
            "  STEP %s | phase=%s step=%s ms=%.2f success=%s detail=%s",
            fname,
            step.get("phase"),
            step.get("step"),
            float(step.get("elapsed_ms") or 0.0),
            step.get("success"),
            step.get("detail"),
        )


def run_pipeline(
    crops_dir: Path,
    results_dir: Path,
    workers: int,
    batch_size: int,
    aggressive_max_passes: int,
) -> int:
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"decode_report_{stamp}.csv"
    log_path = results_dir / f"decode_log_{stamp}.log"
    latest_csv = results_dir / "decode_report_latest.csv"
    latest_log = results_dir / "decode_log_latest.log"

    logger = setup_logger(log_path)
    images = list_crop_images(crops_dir)
    if not images:
        logger.error("No images found in %s", crops_dir)
        return 1

    workers = max(1, workers)
    batch_size = max(1, batch_size)
    batches = chunked(images, batch_size)

    logger.info("Crops dir     : %s", crops_dir)
    logger.info("Results dir   : %s", results_dir)
    logger.info("Images        : %d", len(images))
    logger.info("Workers       : %d", workers)
    logger.info("Batch size    : %d (%d batches)", batch_size, len(batches))
    logger.info("Agg max passes: %d", aggressive_max_passes)
    logger.info("DataMatrix fmt: %s", _DM_FORMAT)
    logger.info("CSV report    : %s", csv_path)
    logger.info("Log file      : %s", log_path)

    all_rows: list[dict[str, Any]] = []
    wall0 = time.perf_counter()

    for bi, batch in enumerate(batches, start=1):
        logger.info("=== Batch %d/%d (%d images) ===", bi, len(batches), len(batch))
        work = [(str(p), aggressive_max_passes) for p in batch]
        t_batch = time.perf_counter()

        if workers == 1:
            batch_rows = [_worker_decode(w) for w in work]
        else:
            batch_rows = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_worker_decode, w): w[0] for w in work}
                for fut in as_completed(futures):
                    path = futures[fut]
                    try:
                        batch_rows.append(fut.result())
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("Worker failed for %s: %s", path, exc)
                        batch_rows.append(
                            asdict(
                                DecodeResult(
                                    filename=Path(path).name,
                                    decoded=False,
                                    method="error",
                                    reason="worker_exception",
                                    error=str(exc),
                                )
                            )
                        )

        # Stable order within batch by filename, with a safe fallback for malformed rows
        batch_rows = [
            {**r, "filename": r.get("filename") or r.get("image_name") or "unknown"}
            for r in batch_rows
        ]
        batch_rows.sort(key=lambda r: (str(r.get("filename") or "unknown"), str(r.get("roi_name") or ""), str(r.get("view_name") or "")))
        for row in batch_rows:
            log_result_steps(logger, row)
            candidate_rows = row.get("candidate_rows") or []
            if candidate_rows:
                for candidate in candidate_rows:
                    candidate_row = dict(candidate)
                    candidate_row["filename"] = row.get("filename") or candidate_row.get("filename") or candidate_row.get("image_name") or "unknown"
                    candidate_row["image_name"] = row.get("filename") or candidate_row.get("image_name") or candidate_row.get("filename") or candidate_row.get("filename") or "unknown"
                    candidate_row["roi_name"] = candidate_row.get("roi_name", "ROI_01")
                    candidate_row["roi_variant"] = candidate_row.get("roi_variant", "")
                    candidate_row["view_name"] = candidate_row.get("view_name", "")
                    all_rows.append(candidate_row)
            else:
                row.setdefault("filename", row.get("image_name") or "unknown")
                row.setdefault("image_name", row.get("filename") or "unknown")
                all_rows.append(row)

        logger.info(
            "Batch %d done in %.2f s | decoded so far %d/%d",
            bi,
            time.perf_counter() - t_batch,
            sum(1 for r in all_rows if bool(r.get("decoded"))),
            len(all_rows),
        )

    wall_s = time.perf_counter() - wall0
    for r in all_rows:
        r.setdefault("filename", r.get("image_name") or "unknown")
        r.setdefault("image_name", r.get("filename") or "unknown")
    all_rows.sort(key=lambda r: (str(r.get("filename") or "unknown"), str(r.get("roi_name") or ""), str(r.get("view_name") or "")))
    write_csv(csv_path, all_rows)
    write_csv(latest_csv, all_rows)

    # Also copy log content pointer via duplicate write of summary into latest log path
    ok = sum(1 for r in all_rows if r["decoded"])
    fail = len(all_rows) - ok
    by_method: dict[str, int] = {}
    for r in all_rows:
        if r["decoded"]:
            by_method[r.get("method") or "unknown"] = by_method.get(r.get("method") or "unknown", 0) + 1

    logger.info("========== SUMMARY ==========")
    logger.info("Total images : %d", len(all_rows))
    logger.info("Decoded OK   : %d", ok)
    logger.info("Failed       : %d", fail)
    logger.info("Wall time    : %.2f s (%.2f img/s)", wall_s, len(all_rows) / wall_s if wall_s else 0.0)
    logger.info("By method    : %s", by_method)
    logger.info("CSV written  : %s", csv_path)
    logger.info("Also         : %s", latest_csv)

    # Mirror log to latest_log
    try:
        latest_log.write_text(log_path.read_text(encoding="utf-8"), encoding="utf-8")
    except OSError:
        pass

    # Console-friendly table
    print("\n--- Decode Report ---")
    print(f"{'FILE':<42} {'OK':<5} {'METHOD':<22} {'TOTAL_MS':>10}  TEXT")
    print("-" * 110)
    for r in all_rows:
        fname = str(r.get("filename") or r.get("image_name") or "unknown")
        print(
            f"{fname:<42} {str(r.get('decoded')):<5} {(r.get('method') or ''):<22} "
            f"{float(r.get('total_ms') or 0):>10.1f}  {(r.get('text') or '')[:40]}"
        )
    print("-" * 110)
    print(f"OK={ok}/{len(all_rows)}  wall={wall_s:.2f}s  csv={csv_path}")

    return 0 if ok > 0 or len(all_rows) == 0 else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cpu = os.cpu_count() or 4
    p = argparse.ArgumentParser(description="Decode DataMatrix crops via variants + ZXing")
    p.add_argument("--crops-dir", type=Path, default=DEFAULT_CROPS_DIR)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--workers", type=int, default=max(1, min(4, cpu)))
    p.add_argument("--batch-size", type=int, default=8, help="Images per multiprocess batch")
    p.add_argument("--aggressive-max-passes", type=int, default=12)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # Windows ProcessPoolExecutor requires freeze_support under some launchers.
    mp.freeze_support()
    args = parse_args(argv)
    return run_pipeline(
        crops_dir=args.crops_dir,
        results_dir=args.results_dir,
        workers=args.workers,
        batch_size=args.batch_size,
        aggressive_max_passes=args.aggressive_max_passes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
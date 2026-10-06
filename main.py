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
import hashlib
import inspect
import logging
import multiprocessing as mp
import os
import re
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
import torch
import zxingcpp

from datamatrix_unet import DataMatrixUNet, predict_probability_map, render_probability_map

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


def _scale_and_pad_view(img: np.ndarray, scale: float, pad_fraction: float) -> np.ndarray:
    """Build a deterministic decode view for quick fallback attempts."""
    out = img
    if scale != 1.0:
        out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    if pad_fraction > 0:
        h, w = out.shape[:2]
        px = max(2, int(w * pad_fraction))
        py = max(2, int(h * pad_fraction))
        out = cv2.copyMakeBorder(out, py, py, px, px, borderType=cv2.BORDER_REPLICATE)

    return out


def _view_signature(img: np.ndarray) -> str:
    """Compact, stable signature to deduplicate repeated decode attempts."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    thumb = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    digest = hashlib.sha1(thumb.tobytes()).hexdigest()
    h, w = gray.shape[:2]
    return f"{w}x{h}:{digest}"


def _rotate_grid_bool(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k)


def _datamatrix_border_score(grid: np.ndarray) -> float:
    """Score how closely a sampled grid matches Data Matrix finder/timing borders."""
    if grid.ndim != 2 or grid.shape[0] < 10 or grid.shape[1] < 10:
        return 0.0

    n = grid.shape[0]
    top = grid[0, :]
    bottom = grid[-1, :]
    left = grid[:, 0]
    right = grid[:, -1]
    alt = (np.arange(n) % 2) == 0

    orientations = [
        (left, bottom, top, right),
        (top, left, right, bottom),
        (right, top, bottom, left),
        (bottom, right, left, top),
    ]

    best = 0.0
    for solid_a, solid_b, alt_a, alt_b in orientations:
        solid_score = float((solid_a.mean() + solid_b.mean()) / 2.0)
        alt_score_a = max(float((alt_a == alt).mean()), float((alt_a != alt).mean()))
        alt_score_b = max(float((alt_b == alt).mean()), float((alt_b != alt).mean()))
        score = 0.55 * solid_score + 0.45 * ((alt_score_a + alt_score_b) / 2.0)
        best = max(best, score)
    return best


def _estimate_datamatrix_finder_pitch(gray: np.ndarray, border_fraction: float = 0.25) -> float:
    """Estimate an approximate module pitch from finder-like border alternation patterns."""
    h, w = gray.shape[:2]
    border_h = max(8, int(h * border_fraction))
    border_w = max(8, int(w * border_fraction))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bw < 128

    candidates: list[float] = []
    for strip in (dark[:border_h, :], dark[:, :border_w], dark[-border_h:, :], dark[:, -border_w:]):
        if strip.size == 0:
            continue
        if strip.shape[0] > strip.shape[1]:
            profile = strip.mean(axis=1)
            crossings = np.where(np.diff(profile > 0.5) != 0)[0]
            if crossings.size >= 2:
                run_lengths = np.diff(crossings)
                valid = run_lengths[run_lengths > 1]
                if valid.size:
                    candidates.append(float(np.median(valid)))
        else:
            profile = strip.mean(axis=0)
            crossings = np.where(np.diff(profile > 0.5) != 0)[0]
            if crossings.size >= 2:
                run_lengths = np.diff(crossings)
                valid = run_lengths[run_lengths > 1]
                if valid.size:
                    candidates.append(float(np.median(valid)))

    if candidates:
        pitch = float(np.median(candidates))
        return max(2.0, min(float(min(h, w)) / 10.0, pitch))
    return max(4.0, min(h, w) / 20.0)


def _estimate_finder_anchor(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    """Estimate a candidate finder region in the top-left/top-right/bottom-left layout."""
    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bw < 128

    regions = [
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (0, h // 2, w // 2, h),
    ]
    best = None
    best_score = -1.0

    for x0, y0, x1, y1 in regions:
        patch = dark[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        density = float(patch.mean())
        if density <= 0.12:
            continue
        yy, xx = np.indices(patch.shape)
        cy, cx = patch.shape[0] / 2.0, patch.shape[1] / 2.0
        dist = np.hypot(yy - cy, xx - cx)
        radial = np.clip(1.0 - dist / max(1.0, max(cy, cx) * 1.5), 0.0, 1.0)
        weighted = float((patch.astype(np.float32) * radial).mean())
        score = density * 0.7 + weighted * 0.3
        if score > best_score:
            best_score = score
            best = (x0, y0, x1, y1)

    return best


def _estimate_finder_centers(gray: np.ndarray) -> list[tuple[float, float]]:
    """Return likely finder corner centers based on corner occupancy and local alternation."""
    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bw < 128

    corner_centers = [
        (w * 0.25, h * 0.25),
        (w * 0.75, h * 0.25),
        (w * 0.25, h * 0.75),
    ]
    scores: list[tuple[float, float, float]] = []

    for cx, cy in corner_centers:
        x0 = max(0, int(cx - w * 0.18))
        x1 = min(w, int(cx + w * 0.18))
        y0 = max(0, int(cy - h * 0.18))
        y1 = min(h, int(cy + h * 0.18))
        patch = dark[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        density = float(patch.mean())
        yy, xx = np.indices(patch.shape)
        cy_p, cx_p = patch.shape[0] / 2.0, patch.shape[1] / 2.0
        dist = np.hypot(yy - cy_p, xx - cx_p)
        radius = max(1.0, min(patch.shape[0], patch.shape[1]) * 0.6)
        radial = np.clip(1.0 - dist / radius, 0.0, 1.0)
        weighted = float((patch.astype(np.float32) * radial).mean())
        score = density * 0.7 + weighted * 0.3
        scores.append((score, float(cx), float(cy)))

    scores.sort(reverse=True)
    return [(cx, cy) for _, cx, cy in scores[:3]]


def _estimate_finder_anchor_candidates(gray: np.ndarray) -> list[dict[str, float]]:
    """Return candidate finder anchors for the top-left/top-right/bottom-left DataMatrix corners."""
    h, w = gray.shape[:2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bw < 128

    quadrants = [
        ("tl", 0.0, 0.0, 0.5, 0.5),
        ("tr", 0.5, 0.0, 1.0, 0.5),
        ("bl", 0.0, 0.5, 0.5, 1.0),
    ]

    candidates: list[dict[str, float]] = []
    for label, x_frac0, y_frac0, x_frac1, y_frac1 in quadrants:
        x0 = int(w * x_frac0)
        y0 = int(h * y_frac0)
        x1 = int(w * x_frac1)
        y1 = int(h * y_frac1)
        patch = dark[y0:y1, x0:x1]
        if patch.size == 0:
            continue

        density = float(patch.mean())
        if density < 0.08:
            continue

        profile_x = patch.mean(axis=0)
        profile_y = patch.mean(axis=1)
        alt_x = np.diff(profile_x > 0.5) != 0
        alt_y = np.diff(profile_y > 0.5) != 0
        alt_score = float((alt_x.mean() + alt_y.mean()) / 2.0)

        yy, xx = np.indices(patch.shape)
        cy, cx = patch.shape[0] / 2.0, patch.shape[1] / 2.0
        dist = np.hypot(yy - cy, xx - cx)
        rad = np.clip(1.0 - dist / max(1.0, min(patch.shape) * 0.8), 0.0, 1.0)
        center_score = float((patch.astype(np.float32) * rad).mean())
        score = 0.5 * density + 0.3 * alt_score + 0.2 * center_score

        candidates.append({
            "label": label,
            "cx": float((x0 + x1) / 2.0),
            "cy": float((y0 + y1) / 2.0),
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "density": density,
            "alt_score": alt_score,
            "center_score": center_score,
            "score": score,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:8]


def _estimate_corner_pitch(gray: np.ndarray, anchor: dict[str, float]) -> float:
    """Estimate a local DataMatrix module pitch from finder-region alternation."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = int(anchor["x0"]), int(anchor["y0"]), int(anchor["x1"]), int(anchor["y1"])
    patch = gray[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if patch.size == 0:
        return max(2.0, min(h, w) / 30.0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(patch)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bw < 128

    run_lengths: list[float] = []
    for profile in (dark.mean(axis=0), dark.mean(axis=1)):
        binary = profile > 0.5
        transitions = np.where(np.diff(binary.astype(np.int8)) != 0)[0]
        if transitions.size < 2:
            continue
        diffs = np.diff(transitions)
        valid = diffs[diffs > 1]
        if valid.size:
            run_lengths.extend(valid.astype(np.float32))

    if run_lengths:
        pitch = float(np.median(run_lengths))
        return max(2.0, min(18.0, pitch))

    return max(2.0, min(h, w) / 25.0)


def _fit_datamatrix_grid(gray: np.ndarray, grid_size: int) -> tuple[np.ndarray, dict[str, float]] | None:
    """Fit a DataMatrix lattice only when the finder geometry and contrast are physically plausible."""
    if gray is None or gray.size == 0 or grid_size < 10:
        return None
    h, w = gray.shape[:2]
    if h < 20 or w < 20:
        return None

    is_tiny_crop = min(h, w) <= 140
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, bw = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = bw < 128

    anchors = _estimate_finder_anchor_candidates(clahe)
    if len(anchors) < 2:
        return None

    best = sorted(anchors, key=lambda a: a["score"], reverse=True)[:3]
    if is_tiny_crop and len(best) < 3:
        return None

    pitches = [_estimate_corner_pitch(clahe, a) for a in best]
    pitch = float(np.median(pitches))
    pitch_min = min(pitches) if pitches else pitch
    pitch_max = max(pitches) if pitches else pitch
    if pitch <= 0.0:
        return None
    if pitch < 2.0 or pitch > max(20.0, max(h, w) / 6.0):
        return None
    if pitch_max / max(pitch_min, 1e-6) > (1.4 if is_tiny_crop else 1.8):
        return None

    cx = np.array([a["cx"] for a in best], dtype=np.float32)
    cy = np.array([a["cy"] for a in best], dtype=np.float32)
    x0 = float(np.min(cx))
    x1 = float(np.max(cx))
    y0 = float(np.min(cy))
    y1 = float(np.max(cy))
    span_x = max(1.0, x1 - x0)
    span_y = max(1.0, y1 - y0)
    if span_x < (12.0 if is_tiny_crop else 10.0) or span_y < (12.0 if is_tiny_crop else 10.0):
        return None

    if is_tiny_crop:
        if span_x < 0.18 * w or span_y < 0.18 * h:
            return None
        dxs = np.abs(cx[:, None] - cx[None, :])
        dys = np.abs(cy[:, None] - cy[None, :])
        pairwise = np.sqrt(dxs ** 2 + dys ** 2)
        pairwise = pairwise[np.triu_indices(len(best), 1)]
        if pairwise.size and float(np.min(pairwise)) < 0.18 * max(w, h):
            return None

    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    module_scores: list[float] = []
    cell_w = span_x / float(grid_size)
    cell_h = span_y / float(grid_size)

    for r in range(grid_size):
        for c in range(grid_size):
            px = x0 + (c + 0.5) * cell_w
            py = y0 + (r + 0.5) * cell_h
            x_start = max(0, int(px - pitch * 0.45))
            x_end = min(w, int(px + pitch * 0.45))
            y_start = max(0, int(py - pitch * 0.45))
            y_end = min(h, int(py + pitch * 0.45))
            if x_end <= x_start or y_end <= y_start:
                grid[r, c] = 0
                continue

            patch = clahe[y_start:y_end, x_start:x_end]
            if patch.size == 0:
                grid[r, c] = 0
                continue

            patch_f = patch.astype(np.float32)
            yy, xx = np.indices(patch.shape)
            cy_patch = patch.shape[0] / 2.0
            cx_patch = patch.shape[1] / 2.0
            dist = np.hypot(yy - cy_patch, xx - cx_patch)
            radius = max(1.0, min(patch.shape) / 2.0)
            weights = np.clip(1.0 - dist / radius, 0.0, 1.0)
            local_mean = float(np.average(patch_f, weights=weights))
            dark_fraction = float((patch_f < 128).mean())
            module_scores.append(local_mean)

            if is_tiny_crop:
                grid[r, c] = 1 if (local_mean < 110 and dark_fraction > 0.25) else 0
            else:
                grid[r, c] = 1 if local_mean < 128 else 0

    if not module_scores:
        return None

    module_scores_arr = np.asarray(module_scores, dtype=np.float32)
    contrast = float(module_scores_arr.std())
    if contrast < (10.0 if not is_tiny_crop else 18.0):
        return None

    border_score = _datamatrix_border_score(grid)
    if border_score < (0.35 if not is_tiny_crop else 0.50):
        return None

    dark_ratio = float(grid.mean())
    if dark_ratio < 0.08 or dark_ratio > 0.92:
        return None

    if is_tiny_crop:
        outer_border = float(np.mean([grid[0, :].mean(), grid[-1, :].mean(), grid[:, 0].mean(), grid[:, -1].mean()]))
        if outer_border < 0.12:
            return None

    meta = {
        "pitch": pitch,
        "border_score": float(border_score),
        "dark_ratio": float(dark_ratio),
        "contrast": float(contrast),
        "finder_count": len(best),
        "pitch_range_ratio": float(pitch_max / max(pitch_min, 1e-6)),
    }
    return grid, meta


def _sample_grid_from_image(img: np.ndarray, grid_size: int) -> tuple[np.ndarray, dict[str, float]] | None:
    """Geometry-driven DataMatrix module fitting with false-positive rejection before decode."""
    if img is None or img.size == 0 or grid_size < 10:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    if gray.shape[0] < 20 or gray.shape[1] < 20:
        return None

    return _fit_datamatrix_grid(gray, grid_size)


def _render_grid_to_bitmap(grid: np.ndarray, module_px: int = 12, quiet_zone: int = 2) -> np.ndarray:
    """Render a sampled boolean grid to a clean synthetic bitmap for ZXing retry."""
    n = int(grid.shape[0])
    side = (n + 2 * quiet_zone) * module_px
    canvas = np.full((side, side), 255, dtype=np.uint8)
    for row in range(n):
        for col in range(n):
            if grid[row, col]:
                y0 = (row + quiet_zone) * module_px
                y1 = y0 + module_px
                x0 = (col + quiet_zone) * module_px
                x1 = x0 + module_px
                canvas[y0:y1, x0:x1] = 0
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def _reconstruct_datamatrix_candidates(roi: np.ndarray) -> list[tuple[str, np.ndarray, float, dict[str, float]]]:
    """Generate only geometry-valid reconstruction candidates from a failed ROI."""
    if roi is None or roi.size == 0:
        return []

    geometry_sources: dict[str, np.ndarray] = {"raw": roi}
    rectified = _rectify_perspective_from_edges(
        roi,
        blur_ksize=3,
        canny_low=30,
        canny_high=120,
        min_area_ratio=0.03,
        approx_eps=0.03,
    )
    if rectified is not None and rectified.size > 0:
        geometry_sources["rectified"] = rectified
    geometry_sources["rot_m15"] = _rotate_image_keep_size(roi, -15)
    geometry_sources["rot_p15"] = _rotate_image_keep_size(roi, 15)

    variant_names = ["original", "grayscale", "clahe", "hist_eq", "adaptive_thresh", "clahe_adaptive_thresh"]
    grid_sizes = [10, 12, 14, 16, 18, 20, 22, 24, 26, 32]

    candidates: list[tuple[str, np.ndarray, float, dict[str, float]]] = []
    seen_renders: set[str] = set()
    for source_name, source_img in geometry_sources.items():
        variants = preprocess_variants(source_img)
        for variant_name in variant_names:
            variant_img = variants.get(variant_name)
            if variant_img is None or variant_img.size == 0:
                continue
            for grid_size in grid_sizes:
                sampled = _sample_grid_from_image(variant_img, grid_size)
                if sampled is None:
                    continue
                grid, meta = sampled

                border_score = float(meta.get("border_score", 0.0))
                if border_score < 0.35:
                    continue
                if meta.get("pitch_range_ratio", 1.0) > 1.8:
                    continue
                dark_ratio = float(meta.get("dark_ratio", 0.0))
                if dark_ratio < 0.08 or dark_ratio > 0.92:
                    continue

                rendered = _render_grid_to_bitmap(grid)
                sig = _view_signature(rendered)
                if sig in seen_renders:
                    continue
                seen_renders.add(sig)
                name = f"reconstruct|{source_name}|{variant_name}|n{grid_size}"
                candidates.append((name, rendered, border_score, meta))

    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates[:12]


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


def _extract_epoch_number(path: Path | str) -> int:
    match = re.search(r"epoch[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else -1


def _find_datamatrix_unet_checkpoint() -> str | None:
    candidate_paths = [
        SCRIPT_DIR / "weights" / "datamatrix_unet_best.pt",
        SCRIPT_DIR / "weights" / "datamatrix_unet.pth",
        SCRIPT_DIR / "datamatrix_unet.pt",
        SCRIPT_DIR / "datamatrix_unet.pth",
        SCRIPT_DIR / "models" / "datamatrix_unet.pt",
    ]
    for path in candidate_paths:
        if path.exists():
            return str(path)

    search_roots = [SCRIPT_DIR / "weights", SCRIPT_DIR, SCRIPT_DIR / "models"]
    epoch_candidates: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("datamatrix_unet_epoch_*.pt", "datamatrix_unet_epoch_*.pth"):
            epoch_candidates.extend(root.glob(pattern))

    if epoch_candidates:
        latest = max(epoch_candidates, key=lambda p: _extract_epoch_number(p))
        return str(latest)

    return None


def _load_datamatrix_unet_model(model_path: str | None = None) -> DataMatrixUNet | None:
    checkpoint = model_path or _find_datamatrix_unet_checkpoint()
    if checkpoint is None:
        return None

    try:
        model = DataMatrixUNet(reasoning_steps=3)
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model_state_dict" in state:
                state = state["model_state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        return model
    except Exception:
        return None


def _datamatrix_unet_reconstruction_candidates(roi: np.ndarray) -> list[tuple[str, np.ndarray, float, dict[str, float]]]:
    """Try a trained DataMatrix U-Net reconstruction as an additional candidate source."""
    if roi is None or roi.size == 0:
        return []

    model = _load_datamatrix_unet_model()
    if model is None:
        return []

    try:
        prob = predict_probability_map(model, roi)
        binary = (prob > 0.5).astype(np.uint8)
        if binary.size == 0:
            return []

        rendered = render_probability_map(prob, module_px=12, quiet_zone=2)
        score = float(_barcode_likeness_score(rendered))
        border_score = float(0.5 + 0.5 * score)
        dark_ratio = float(binary.mean())
        if dark_ratio <= 0.05 or dark_ratio >= 0.95:
            return []

        return [
            (
                "reconstruct|unet|prob05",
                rendered,
                border_score,
                {"border_score": border_score, "pitch_range_ratio": 1.0, "dark_ratio": dark_ratio},
            )
        ]
    except Exception:
        return []


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

    # High-yield quick rescue sequence from empirical recovery runs.
    quick_fallback_recipes = [
        ("upscale_1.5_pad_0.05", 1.5, 0.05),
        ("upscale_3.0_pad_0.0", 3.0, 0.0),
        ("upscale_1.5_pad_0.0", 1.5, 0.0),
    ]

    seen_attempt_keys: set[tuple[str, str]] = set()
    seen_candidate_rows: set[tuple[str, str, int, int, float, float]] = set()

    no_score_passes = 0

    # Stage 0: quick fallback before expensive geometry sweeps.
    for recipe_name, scale, pad_fraction in quick_fallback_recipes:
        quick_view = _scale_and_pad_view(roi, scale=scale, pad_fraction=pad_fraction)
        quick_metrics = _image_quality_metrics(quick_view)
        row_key = (
            "quick",
            recipe_name,
            int(quick_view.shape[1]),
            int(quick_view.shape[0]),
            round(float(quick_metrics["barcode_score"]), 4),
            round(float(quick_metrics["quiet_zone_score"]), 4),
        )
        if row_key not in seen_candidate_rows:
            seen_candidate_rows.add(row_key)
            candidate_rows.append(
                {
                    "image_name": "",
                    "roi_name": "ROI_01",
                    "roi_variant": f"quick|{recipe_name}",
                    "view_name": f"quick|{recipe_name}|native",
                    "decoded": False,
                    "text": "",
                    "format": "",
                    "method": "aggressive_fallback",
                    "reason": "candidate_evaluated",
                    "contrast": quick_metrics["contrast"],
                    "lap_var": quick_metrics["lap_var"],
                    "quiet_zone_score": quick_metrics["quiet_zone_score"],
                    "glare_border_ratio": quick_metrics["glare_border_ratio"],
                    "edge_density": quick_metrics["edge_density"],
                    "barcode_score": quick_metrics["barcode_score"],
                    "pass_idx": 0,
                    "rotation_deg": 0,
                    "roi_w": int(quick_view.shape[1]),
                    "roi_h": int(quick_view.shape[0]),
                }
            )

        for override in fallback_overrides[:2]:
            override_key = str(sorted(override.items()))
            attempt_key = (_view_signature(quick_view), override_key)
            if attempt_key in seen_attempt_keys:
                continue
            seen_attempt_keys.add(attempt_key)

            t_try = time.perf_counter()
            decoded = _decode_with_config(quick_view, override=override, base_config=base_cfg)
            elapsed = (time.perf_counter() - t_try) * 1000.0
            label = f"quick|{recipe_name}|{override.get('binarizer', 'default')}"
            if decoded:
                for d in decoded:
                    txt = (d.text or "").strip()
                    if not txt:
                        continue
                    steps.append(
                        StepLog(
                            phase="aggressive",
                            step=label,
                            elapsed_ms=elapsed,
                            success=True,
                            detail=f"text={txt[:80]}",
                        )
                    )
                    candidate_rows.append(
                        {
                            "image_name": "",
                            "roi_name": "ROI_01",
                            "roi_variant": f"quick|{recipe_name}",
                            "view_name": f"quick|{recipe_name}|native",
                            "decoded": True,
                            "text": txt,
                            "format": str(getattr(d, "format", "DataMatrix")),
                            "method": "aggressive_fallback",
                            "reason": "barcode_decoded",
                            "contrast": quick_metrics["contrast"],
                            "lap_var": quick_metrics["lap_var"],
                            "quiet_zone_score": quick_metrics["quiet_zone_score"],
                            "glare_border_ratio": quick_metrics["glare_border_ratio"],
                            "edge_density": quick_metrics["edge_density"],
                            "barcode_score": quick_metrics["barcode_score"],
                            "pass_idx": 0,
                            "rotation_deg": 0,
                            "roi_w": int(quick_view.shape[1]),
                            "roi_h": int(quick_view.shape[0]),
                        }
                    )
                    return {
                        "decoded": True,
                        "text": txt,
                        "format": str(getattr(d, "format", "DataMatrix")),
                        "variant": label,
                        "attempts": 0,
                        "steps": steps,
                        "reason": "decoded_quick_fallback",
                        "candidate_rows": candidate_rows,
                    }

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

            # Skip repeated candidates that do not add new value to logs/CSV.
            candidate_row_key = (
                crop_name,
                f"{crop_name}|native",
                int(crop_img.shape[1]),
                int(crop_img.shape[0]),
                round(float(metrics["barcode_score"]), 4),
                round(float(metrics["quiet_zone_score"]), 4),
            )
            if candidate_row_key in seen_candidate_rows:
                continue
            seen_candidate_rows.add(candidate_row_key)

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
                    override_key = str(sorted(override.items()))
                    attempt_key = (_view_signature(view), override_key)
                    if attempt_key in seen_attempt_keys:
                        continue
                    seen_attempt_keys.add(attempt_key)

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


def _reconstruction_decode(roi: np.ndarray, steps: list[StepLog]) -> dict[str, Any]:
    """Post-failure constrained reconstruction stage for weak but structured Data Matrix crops."""
    base_cfg = {
        "formats": _DM_FORMAT,
        "try_rotate": True,
        "try_downscale": False,
        "try_harder": True,
        "binarizer": "LOCAL_AVERAGE",
        "text_mode": None,
    }

    candidate_rows: list[dict[str, Any]] = []
    recon_candidates = _reconstruct_datamatrix_candidates(roi)
    unet_candidates = _datamatrix_unet_reconstruction_candidates(roi)
    if unet_candidates:
        recon_candidates = unet_candidates + recon_candidates
    steps.append(
        StepLog(
            phase="reconstruct",
            step="generate_candidates",
            elapsed_ms=0.0,
            success=bool(recon_candidates),
            detail=f"count={len(recon_candidates)}",
        )
    )

    if not recon_candidates:
        metrics = _image_quality_metrics(roi)
        candidate_rows.append(
            {
                "image_name": "",
                "roi_name": "ROI_01",
                "roi_variant": "reconstruct|none",
                "view_name": "reconstruct|none|native",
                "decoded": False,
                "text": "",
                "format": "",
                "method": "reconstruction_fallback",
                "reason": "reconstruction_no_candidates",
                "contrast": metrics["contrast"],
                "lap_var": metrics["lap_var"],
                "quiet_zone_score": metrics["quiet_zone_score"],
                "glare_border_ratio": metrics["glare_border_ratio"],
                "edge_density": metrics["edge_density"],
                "barcode_score": metrics["barcode_score"],
                "pass_idx": 0,
                "rotation_deg": 0,
                "roi_w": int(roi.shape[1]),
                "roi_h": int(roi.shape[0]),
            }
        )

    for idx, (name, image, border_score, meta) in enumerate(recon_candidates, start=1):
        metrics = _image_quality_metrics(image)
        candidate_rows.append(
            {
                "image_name": "",
                "roi_name": "ROI_01",
                "roi_variant": name,
                "view_name": f"{name}|native",
                "decoded": False,
                "text": "",
                "format": "",
                "method": "reconstruction_fallback",
                "reason": "reconstruction_candidate",
                "contrast": metrics["contrast"],
                "lap_var": metrics["lap_var"],
                "quiet_zone_score": metrics["quiet_zone_score"],
                "glare_border_ratio": metrics["glare_border_ratio"],
                "edge_density": metrics["edge_density"],
                "barcode_score": metrics["barcode_score"],
                "pass_idx": idx,
                "rotation_deg": 0,
                "roi_w": int(image.shape[1]),
                "roi_h": int(image.shape[0]),
            }
        )

        t_try = time.perf_counter()
        decoded = _decode_with_config(image, base_config=base_cfg)
        elapsed = (time.perf_counter() - t_try) * 1000.0
        if decoded:
            for d in decoded:
                txt = (d.text or "").strip()
                if not txt:
                    continue
                steps.append(
                    StepLog(
                        phase="reconstruct",
                        step=name,
                        elapsed_ms=elapsed,
                        success=True,
                        detail=f"border_score={border_score:.3f} text={txt[:80]}",
                    )
                )
                candidate_rows.append(
                    {
                        "image_name": "",
                        "roi_name": "ROI_01",
                        "roi_variant": name,
                        "view_name": f"{name}|native",
                        "decoded": True,
                        "text": txt,
                        "format": str(getattr(d, "format", "DataMatrix")),
                        "method": "reconstruction_fallback",
                        "reason": "decoded_reconstruction",
                        "contrast": metrics["contrast"],
                        "lap_var": metrics["lap_var"],
                        "quiet_zone_score": metrics["quiet_zone_score"],
                        "glare_border_ratio": metrics["glare_border_ratio"],
                        "edge_density": metrics["edge_density"],
                        "barcode_score": metrics["barcode_score"],
                        "pass_idx": idx,
                        "rotation_deg": 0,
                        "roi_w": int(image.shape[1]),
                        "roi_h": int(image.shape[0]),
                    }
                )
                return {
                    "decoded": True,
                    "text": txt,
                    "format": str(getattr(d, "format", "DataMatrix")),
                    "variant": name,
                    "attempts": idx,
                    "steps": steps,
                    "reason": "decoded_reconstruction",
                    "candidate_rows": candidate_rows,
                }

        steps.append(
            StepLog(
                phase="reconstruct",
                step=name,
                elapsed_ms=elapsed,
                success=False,
                detail=f"border_score={border_score:.3f}",
            )
        )

    return {
        "decoded": False,
        "text": "",
        "format": "",
        "variant": "",
        "attempts": len(recon_candidates),
        "steps": steps,
        "reason": "reconstruction_failed",
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
            t_reconstruct = time.perf_counter()
            reconstructed = _reconstruction_decode(img, steps)
            result.aggressive_ms += (time.perf_counter() - t_reconstruct) * 1000.0
            steps = reconstructed["steps"]
            result.candidate_rows.extend(reconstructed.get("candidate_rows", []))

            if reconstructed["decoded"]:
                result.decoded = True
                result.text = reconstructed["text"]
                result.format = reconstructed["format"]
                result.method = "reconstruction_fallback"
                result.variant = reconstructed["variant"]
                result.reason = reconstructed.get("reason", "decoded_reconstruction")
            else:
                result.method = "none"
                result.reason = reconstructed.get("reason", aggressive.get("reason", "failed"))

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
    if crops_dir.is_file():
        return [crops_dir] if crops_dir.suffix.lower() in IMAGE_EXTS else []

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
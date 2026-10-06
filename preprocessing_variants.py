import cv2
import numpy as np
import zxingcpp
import inspect 
from ultralytics import YOLO


def cylindrical_unwarp_and_threshold(roi):
    """
    Corrected mathematical cylindrical unwrapping for curved surfaces (e.g., bottles, cans).
    Maps pixel points based on cylinder radius to eliminate horizontal compression.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    
    # 1. Establish the mid-axis and calculate a practical cylinder radius estimate
    cx = w / 2.0
    # A safe assumption radius: slightly larger than half the width
    R = w * 0.75  

    # 2. Build map coordinates
    yy, xx = np.indices((h, w), dtype=np.float32)
    
    # Distance from vertical cylinder centerline axis
    dx = xx - cx
    
    # Inverse mapping formula: x_projected = R * arcsin(dx / R)
    # Using np.clip to prevent NaN errors on edge boundary math
    sin_angle = np.clip(dx / R, -0.999, 0.999)
    unwrapped_x = cx + R * np.arcsin(sin_angle)
    unwrapped_y = yy

    # 3. Remap coordinates to flatten the curved surface grid
    unwrapped = cv2.remap(
        gray,
        unwrapped_x,
        unwrapped_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # 4. Standardize contrast & binarization for 2D/1D scanning
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7, 7)).apply(unwrapped)
    _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)


def add_quiet_zone(binary_gray_img, padding=15):
    """
    Helper to inject a solid white border around high-threshold variants.
    Prevents zxingcpp from failing due to edge-bleeding backgrounds.
    """
    return cv2.copyMakeBorder(
        binary_gray_img, padding, padding, padding, padding,
        cv2.BORDER_CONSTANT, value=255
    )


def preprocess_variants(img):
    """
    Generate multiple preprocessed versions of an image to maximize
    barcode decode success across varying backgrounds and intensities.
    Returns a dict of BGR images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_sharpen = np.array([[0, -1,  0],
                               [-1,  5, -1],
                               [ 0, -1,  0]])

    # HSV: separate hue/saturation from brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # LAB: L channel = pure luminance, independent of color
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    variants = {
        # ── Grayscale-based ───────────────────────────────────────────────────
        'original'        : img,
        'grayscale'       : cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        'hist_eq'         : cv2.cvtColor(cv2.equalizeHist(gray), cv2.COLOR_GRAY2BGR),
        'gaussian_blur'   : cv2.cvtColor(cv2.GaussianBlur(gray, (3, 3), 0), cv2.COLOR_GRAY2BGR),
        'inverted'        : cv2.cvtColor(cv2.bitwise_not(gray), cv2.COLOR_GRAY2BGR),
        'sharpened'       : cv2.cvtColor(cv2.filter2D(gray, -1, kernel_sharpen), cv2.COLOR_GRAY2BGR),
        'adaptive_thresh' : cv2.cvtColor(
                               add_quiet_zone(cv2.adaptiveThreshold(
                                   gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 8)),
                               cv2.COLOR_GRAY2BGR),
        'clahe'           : cv2.cvtColor(
                               cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6, 6)).apply(gray),
                               cv2.COLOR_GRAY2BGR),

        # ── Color-aware (handles colored backgrounds) ─────────────────────────
        # HSV Value channel: pure brightness, strips color noise from background
        'hsv_value'       : cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR),

        # HSV Saturation channel: colorful backgrounds show high saturation,
        # black/white barcodes show low saturation — useful contrast separator
        'hsv_saturation'  : cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR),

        # LAB L-channel: perceptual luminance, better than grayscale for
        # separating similar-brightness colors (e.g. red bars on green bg)
        'lab_luminance'   : cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR),
        
        # Geometrically adjusted option for bottles/cans
        'cylindrical_unwrap': cylindrical_unwarp_and_threshold(img),
    }
    
    # ── Advanced Compounded Thresholding ──────────────────────────────────
    clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)

    th = cv2.adaptiveThreshold(
        clahe_obj, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2
    )
    variants['clahe_adaptive_thresh'] = cv2.cvtColor(add_quiet_zone(th), cv2.COLOR_GRAY2BGR)
    
    # Custom grayscale-safe edge enhancer for thin / partially masked barcodes
    gray_for_variant = gray.copy()
    gaussian_blur = cv2.GaussianBlur(gray_for_variant, (0, 0), 3.0)
    sharpened = cv2.addWeighted(gray_for_variant, 1.8, gaussian_blur, -0.8, 0)

    clahe_final = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    enhanced = clahe_final.apply(sharpened)

    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    repaired = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    variants['unmask_morpholy_edge_enhancer'] = cv2.cvtColor(
        add_quiet_zone(repaired),
        cv2.COLOR_GRAY2BGR,
    )

    # ── Integrated 2D Matrix Filtering (Edge-Preserving Denoise + Otsu) ───
    # Local Contrast Correction
    clahe_2d = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted_2d = clahe_2d.apply(gray)
    
    # Edge-Preserving Denoise to protect crisp QR/DataMatrix module bounds
    filtered_2d = cv2.bilateralFilter(contrasted_2d, d=5, sigmaColor=50, sigmaSpace=50)
    
    # Global Otsu Thresholding
    _, binary_2d = cv2.threshold(filtered_2d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Inject a clean 20px solid white Quiet Zone required by 2D engines
    padded_2d = cv2.copyMakeBorder(
        binary_2d, 20, 20, 20, 20, 
        cv2.BORDER_CONSTANT, value=255
    )
    variants['optimized_2d_matrix'] = cv2.cvtColor(padded_2d, cv2.COLOR_GRAY2BGR)
    
    return variants

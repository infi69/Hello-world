
import cv2
import numpy as np
import zxingcpp
import inspect 
from ultralytics import YOLO


def cylindrical_unwarp_and_threshold(roi):
    """
    Approximate cylindrical unwrapping for curved barcode ROIs.
    This maps the curved surface into a flatter strip and then enhances
    the black/white barcode pattern.
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    # Build angle-based mapping to unwrap a cylindrical surface.
    yy, xx = np.indices((h, w), dtype=np.float32)
    dx = xx - cx
    dy = yy - cy

    # angle around center; this acts like unwrapping around a cylinder
    theta = np.arctan2(dy, dx + 1e-6)

    # Map angle back to horizontal coordinate so the curved surface looks flat
    unwrapped_x = ((theta + np.pi) / (2 * np.pi)) * (w - 1)
    unwrapped_y = yy

    # Warp the ROI into a pseudo-flat view
    unwrapped = cv2.remap(
        gray,
        unwrapped_x,
        unwrapped_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Improve contrast before zxing
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7 )).apply(unwrapped)
    _, thresh = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Optional: stronger enhancement for hard-to-read prints
    # thresh = cv2.adaptiveThreshold(
    #     clahe, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     31, 10
    # )

    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    
def preprocess_variants(img):
    """
    Generate multiple preprocessed versions of an image to maximize
    barcode decode success across varying backgrounds and intensities.
    Returns a dict of BGR images (zxingcpp expects BGR or grayscale as 3ch).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_sharpen = np.array([[0, -1,  0],
                              [-1,  5, -1],
                              [ 0, -1,  0]])

    #HSV: separate hue/saturation from brightness
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
                               cv2.adaptiveThreshold(
                                   gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 8),
                               cv2.COLOR_GRAY2BGR),
        'clahe'           : cv2.cvtColor(
                               cv2.createCLAHE(clipLimit=2.5, tileGridSize=(6,6)).apply(gray),
                               cv2.COLOR_GRAY2BGR),

        # ── Color-aware (handles colored backgrounds) ─────────────────────────
        #HSV Value channel: pure brightness, strips color noise from background
        'hsv_value'       : cv2.cvtColor(hsv[:, :, 2], cv2.COLOR_GRAY2BGR),

        # HSV Saturation channel: colorful backgrounds show high saturation,
        # black/white barcodes show low saturation — useful contrast separator
        'hsv_saturation'  : cv2.cvtColor(hsv[:, :, 1], cv2.COLOR_GRAY2BGR),

        # LAB L-channel: perceptual luminance, better than grayscale for
        # separating similar-brightness colors (e.g. red bars on green bg)
        'lab_luminance'   : cv2.cvtColor(lab[:, :, 0], cv2.COLOR_GRAY2BGR),
        'cylindrical_unwrap': cylindrical_unwarp_and_threshold(img),
    }
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6)).apply(gray)

    th = cv2.adaptiveThreshold(
        clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        2
    )

    variants['clahe_adaptive_thresh'] = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
    return variants



    

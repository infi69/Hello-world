import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def layer1_tiled_yolo_localize(
    img_path,
    OUTPUT_DIR,
    img=None,
    tile_size=320,
    overlap=0.4,
    conf_thresh=0.6,
    nms_iou_thresh=0.5,
    yolo_iou_thresh=0.1,     # CHANGED: exposed YOLO per-tile NMS IoU as a parameter
    pad_ratio=0.10,          # CHANGED: configurable ROI padding ratio
    min_pad_px=4,            # CHANGED: minimum padding in pixels
    min_box_side_px=8,       # CHANGED: skip tiny/invalid detections before and after NMS
):
    """
    Layer 1: Tile the image, run YOLO on each raw tile,
    apply NMS to remove overlaps, and save as localized_<name>.jpeg.

    Returns:
        localized_img : annotated image with detected regions drawn
        roi_boxes     : list of dicts with keys 'bbox', 'polygon', 'conf'
    """
    print("Loading YOLO model...")
    yolo_model = YOLO("YOLOV8s_Barcode_Detection.pt")

    if img is None:
        img = cv2.imread(str(img_path))
    if img is None:
        return None, []

    h, w = img.shape[:2]

    # CHANGED: guard overlap and tile_size to avoid degenerate stepping
    tile_size = max(32, int(tile_size))
    overlap = float(np.clip(overlap, 0.0, 0.95))
    step = max(1, int(tile_size * (1.0 - overlap)))

    all_boxes = []
    all_confs = []

    for ty in range(0, h, step):
        for tx in range(0, w, step):
            tx2, ty2 = min(tx + tile_size, w), min(ty + tile_size, h)
            tile = img[ty:ty2, tx:tx2]
            if tile is None or tile.size == 0:
                continue

            # Run YOLO only on raw tile
            yolo_results = yolo_model.predict(
                source=tile,
                conf=conf_thresh,
                iou=yolo_iou_thresh,   # CHANGED
                augment=False,
                verbose=False,
            )

            for r in yolo_results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue

                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])

                    # CHANGED: clamp local coords to tile bounds
                    bx1 = max(0, min(bx1, tile.shape[1] - 1))
                    by1 = max(0, min(by1, tile.shape[0] - 1))
                    bx2 = max(0, min(bx2, tile.shape[1]))
                    by2 = max(0, min(by2, tile.shape[0]))

                    bw = bx2 - bx1
                    bh = by2 - by1
                    if bw < min_box_side_px or bh < min_box_side_px:
                        continue  # CHANGED: ignore tiny detections

                    ox1, oy1 = tx + bx1, ty + by1
                    ox2, oy2 = tx + bx2, ty + by2

                    all_boxes.append([ox1, oy1, ox2 - ox1, oy2 - oy1])
                    all_confs.append(float(box.conf[0]))

    roi_boxes = []
    if all_boxes:
        indices = cv2.dnn.NMSBoxes(
            all_boxes,
            all_confs,
            score_threshold=conf_thresh,
            nms_threshold=nms_iou_thresh,
        )

        # CHANGED: robust handling when OpenCV returns empty tuple/list/array
        if indices is not None and len(indices) > 0:
            for i in np.array(indices).reshape(-1):
                bx, by, bw, bh = all_boxes[int(i)]

                if bw < min_box_side_px or bh < min_box_side_px:
                    continue  # CHANGED: post-NMS tiny box guard

                ox1, oy1, ox2, oy2 = bx, by, bx + bw, by + bh

                # CHANGED: configurable padding with minimum pixels
                pad_x = max(min_pad_px, int(bw * pad_ratio))
                pad_y = max(min_pad_px, int(bh * pad_ratio))

                ox1 = max(0, ox1 - pad_x)
                oy1 = max(0, oy1 - pad_y)
                ox2 = min(w, ox2 + pad_x)
                oy2 = min(h, oy2 + pad_y)

                # CHANGED: final validity guard
                if (ox2 - ox1) < min_box_side_px or (oy2 - oy1) < min_box_side_px:
                    continue

                roi_boxes.append({
                    "bbox": (ox1, oy1, ox2, oy2),
                    "polygon": [(ox1, oy1), (ox2, oy1), (ox2, oy2), (ox1, oy2)],
                    "conf": all_confs[int(i)],
                })

    localized_img = img.copy()
    for rb in roi_boxes:
        x1, y1, x2, y2 = rb["bbox"]
        cv2.rectangle(localized_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            localized_img,
            f"{rb['conf']:.2f}",
            (x1, max(y1 - 6, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    out_path = OUTPUT_DIR / f"localized_{Path(img_path).stem}.jpeg"
    cv2.imwrite(str(out_path), localized_img)
    print(f"  Saved localized image ({len(roi_boxes)} ROI(s)): {out_path}")

    return localized_img, roi_boxes
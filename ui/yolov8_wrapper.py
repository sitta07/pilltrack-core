# ui/yolov8_wrapper.py
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class YOLODetector:
    """
    YOLOv8 Wrapper for detection + segmentation + crop objects
    """

    def __init__(self, model_path="yolov8n-seg.pt", device="cpu"):
        """
        model_path: path to .pt YOLOv8 segmentation model
        device: 'cpu' or 'cuda'
        """
        self.model = YOLO(model_path)
        self.model.fuse()  # optional, for speed
        self.device = device

    def predict(self, frame, conf=0.3):
        """
        Predict object in frame
        frame: BGR numpy array
        conf: confidence threshold
        return: first box, mask (or None)
        """
        results = self.model(frame, imgsz=640, conf=conf, device=self.device)

        if len(results) == 0 or len(results[0].boxes) == 0:
            return None, None

        box = results[0].boxes[0]  # take first detected object
        mask = results[0].masks.data[0] if results[0].masks is not None else None
        return box, mask

    def cutout_by_mask(self, frame, box, mask):
        """
        Crop object using mask
        frame: BGR numpy array
        box: YOLO Box object
        mask: YOLO mask tensor
        return: cropped object BGR image
        """
        if mask is None:
            # fallback: crop by bounding box
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            return frame[y1:y2, x1:x2]

        # convert mask to uint8 binary
        mask = mask.cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255  # binary mask 0/255

        # resize mask to match frame size if needed
        mask_h, mask_w = mask.shape
        h, w = frame.shape[:2]
        if (mask_h != h) or (mask_w != w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # apply mask
        obj = cv2.bitwise_and(frame, frame, mask=mask)

        # crop bounding box
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        # clamp coordinates to frame size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return obj[y1:y2, x1:x2]

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class ImageProcessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def apply_filters(self, frame, bright=0, contrast=1.0, preset="Default"):
        """จัดการแค่แสงและสีเท่านั้น ไม่ยุ่งกับขนาดภาพ (No Zoom)"""
        if frame is None: return None
        
        # 1. Preset (Texture Enhancement)
        if preset == "Pill Enhanced (Texture)":
            frame = self._enhance_pill_texture(frame)

        # 2. Brightness/Contrast 
        # (ทำแค่ตอนที่ค่าเปลี่ยนเพื่อประหยัด CPU)
        if bright != 0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        
        return frame

    def cutout_by_mask(self, original_frame, box, mask_tensor):
        """ใช้ตัดพื้นหลังเหมือนเดิม (ส่วนนี้ไม่เกี่ยวกับการซูมจอกลาง)"""
        if box is None or mask_tensor is None: return None

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        h_img, w_img = original_frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
        
        if x2 - x1 <= 0 or y2 - y1 <= 0: return None

        # Resize mask ให้เท่าภาพจริง
        mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(mask_expanded, size=(h_img, w_img), mode='bilinear').squeeze()

        # ตัดเฉพาะส่วนเม็ดยา
        roi_mask = mask_resized[y1:y2, x1:x2].cpu().numpy()
        roi_image = original_frame[y1:y2, x1:x2]
        binary_mask = (roi_mask > 0.5).astype(np.uint8)

        # กันเหนียวเรื่องขนาด 1px
        if binary_mask.shape[:2] != roi_image.shape[:2]:
            binary_mask = cv2.resize(binary_mask, (roi_image.shape[1], roi_image.shape[0]))

        return cv2.bitwise_and(roi_image, roi_image, mask=binary_mask)

    def _enhance_pill_texture(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
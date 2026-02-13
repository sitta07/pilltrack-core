import cv2
import numpy as np
import torch
import torch.nn.functional as F

class ImageProcessor:
    def __init__(self):
        # ยังคงเก็บค่าสีเขียวไว้เผื่อพี่อยาก switch กลับมาใช้แบบเดิม
        self.lower_green = np.array([35, 40, 40])
        self.upper_green = np.array([90, 255, 255])
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def apply_filters(self, frame, zoom=1.0, bright=0, contrast=1.0, preset="Default"):
        """จัดการ Zoom และแสงสี"""
        if frame is None: return None
        
        # 1. Preset (Texture Enhancement)
        if preset == "Pill Enhanced (Texture)":
            frame = self._enhance_pill_texture(frame)
            
        # 2. Digital Zoom (High Quality Center Crop)
        if zoom > 1.0:
            h, w = frame.shape[:2]
            new_w = int(w / zoom)
            new_h = int(h / zoom)
            
            center_x = w // 2
            center_y = h // 2
            
            x1 = max(0, center_x - (new_w // 2))
            y1 = max(0, center_y - (new_h // 2))
            
            cropped = frame[y1:y1+new_h, x1:x1+new_w]
            # ใช้ LANCZOS4 เพื่อความคมชัดสูงสุดเวลาขยาย
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

        # 3. Brightness/Contrast
        if bright != 0 or contrast != 1.0:
            frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=bright)
        
        return frame

    def cutout_by_mask(self, original_frame, box, mask_tensor):
        """
        🔥 Highlight: ตัดพื้นหลังโดยใช้ AI Mask (GPU Accelerated)
        """
        # Safety First: ถ้าไม่มี Box หรือ Mask (เช่นโหมด Detection) ให้ Return None หรือภาพเปล่าๆ
        if box is None or mask_tensor is None:
            return None

        # 1. ดึงพิกัดกล่อง (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        
        # ป้องกันค่าติดลบหรือเกินขอบภาพ
        h_img, w_img = original_frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)
        
        # ถ้ากล่องมีขนาดผิดปกติ ให้ข้าม
        if x2 - x1 <= 0 or y2 - y1 <= 0: return None

        # 2. Process Mask (แก้ Error ตรงนี้!)
        # ✅ Must Convert to Float: bilinear ต้องการ float เท่านั้น
        mask_float = mask_tensor.float() 

        # เพิ่ม dimension เพื่อ interpolate (N, C, H, W)
        mask_expanded = mask_float.unsqueeze(0).unsqueeze(0) 
        
        # Resize Mask ให้เท่ากับขนาดภาพจริง
        mask_resized = F.interpolate(
            mask_expanded, 
            size=(h_img, w_img), 
            mode='bilinear', 
            align_corners=False
        ).squeeze() # เอา dimension ที่เกินออก

        # 3. ตัดเฉพาะส่วนกล่อง (ROI)
        # ใช้ .cpu().numpy() เพื่อแปลงกลับมาใช้งานกับ OpenCV
        roi_mask = mask_resized[y1:y2, x1:x1+(x2-x1)].cpu().numpy()
        roi_image = original_frame[y1:y2, x1:x2]

        # เช็คขนาดอีกทีกันเหนียว (บางทีการปัดเศษทำให้ size ไม่เท่ากันเป๊ะ)
        mh, mw = roi_mask.shape
        ih, iw = roi_image.shape[:2]
        
        if mh != ih or mw != iw:
            # Resize mask ให้เท่ากับ image roi เป๊ะๆ
            roi_mask = cv2.resize(roi_mask, (iw, ih), interpolation=cv2.INTER_NEAREST)

        # ทำให้เป็น Binary Mask (0 หรือ 255)
        # แนะนำให้ใช้ 255 ไปเลยครับ ชัวร์กว่ากับ OpenCV
        binary_mask = (roi_mask > 0.5).astype(np.uint8) * 255

        # 4. Apply Mask! (พื้นหลังจะเป็นสีดำ)
        result = cv2.bitwise_and(roi_image, roi_image, mask=binary_mask)
        
        return result

    def _enhance_pill_texture(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        merged = cv2.merge((l, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def draw_crosshair(self, frame):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        length, gap = 20, 5
        color = (0, 255, 0) # Green
        
        cv2.line(frame, (cx - length, cy), (cx - gap, cy), color, 2)
        cv2.line(frame, (cx + gap, cy), (cx + length, cy), color, 2)
        cv2.line(frame, (cx, cy - length), (cx, cy - gap), color, 2)
        cv2.line(frame, (cx, cy + gap), (cx, cy + length), color, 2)
        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1) 
        return frame
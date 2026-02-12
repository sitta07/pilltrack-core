import os
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        # ถ้าพี่อยากได้ละเอียดกริบๆ ให้เปลี่ยน n เป็น s หรือ m (yolov8s-seg.pt)
        # แต่ถ้าเอาเร็วจัดๆ ใช้ n เหมือนเดิม (yolov8n-seg.pt)
        seg_model_path = model_path.replace('.pt', '-seg.pt') # บังคับใช้ตัว seg
        
        try:
            # task='segment' คือหัวใจสำคัญ
            self.model = YOLO(seg_model_path, task='segment') 
            return True, f"Loaded SEGMENTATION model: {seg_model_path}"
        except Exception as e:
            # ถ้าไม่มีไฟล์ เดี๋ยว Ultralytics มันโหลดให้เอง
            try:
                self.model = YOLO('yolov8n-seg.pt', task='segment')
                return True, "Loaded Default yolov8n-seg.pt"
            except Exception as e2:
                return False, str(e2)

    def predict(self, frame, conf=0.1):
        if self.model is None: return None, None
        
        # device=0 คือบังคับใช้ RTX 5060 Ti ของพี่
        results = self.model.predict(
            source=frame, 
            conf=conf, 
            verbose=False, 
            device=0, 
            retina_masks=True # 🔥 เปิดตัวนี้เพื่อ Mask คมกริบ (กิน GPU นิดนึงแต่พี่ไหวอยู่แล้ว)
        )
        
        if results and results[0].boxes:
            # หาตัวที่ Conf สูงสุด
            best_idx = torch.argmax(results[0].boxes.conf).item()
            
            box = results[0].boxes[best_idx]
            
            # ดึง Mask ออกมา (ถ้ามี)
            mask = None
            if results[0].masks is not None:
                # masks.data จะได้ mask ของทั้งภาพ
                mask = results[0].masks.data[best_idx]
                
            return box, mask
            
        return None, None
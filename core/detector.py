import os
import torch
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        # 1. แก้ Logic การโหลดไฟล์: ไม่บังคับแก้ชื่อไฟล์เป็น -seg.pt แล้ว
        # เพื่อให้รองรับ .onnx หรือไฟล์ชื่ออื่นๆ ที่ส่งเข้ามาตรงๆ
        
        try:
            # task='detect' คือหัวใจสำคัญสำหรับ yolov8n.onnx (Detection)
            # ถ้าโหลด onnx ไม่ต้องระบุ device ตรงนี้ Ultralytics จะจัดการเอง
            self.model = YOLO(model_path, task='detect') 
            return True, f"Loaded DETECTION model: {model_path}"
        except Exception as e:
            # Fallback: ถ้าโหลดไม่ได้ ให้กลับไปใช้ default
            try:
                print(f"Error loading {model_path}: {e}, switching to default.")
                self.model = YOLO('yolov8n.onnx', task='detect')
                return True, "Loaded Default yolov8n.onnx"
            except Exception as e2:
                return False, str(e2)

    def predict(self, frame, conf=0.1):
        if self.model is None: return None, None
        
        # 2. แก้ Predict: ตัด retina_masks ออก เพราะ ONNX Detection ไม่ใช้
        # device=0 ถ้าใช้ ONNX Runtime GPU ได้มันจะใช้เอง แต่ถ้าไม่มีมันจะวิ่ง CPU
        results = self.model.predict(
            source=frame, 
            conf=conf, 
            verbose=False, 
            device=0,
            # retina_masks=True  <-- ลบบรรทัดนี้ออก เพราะ Detection ไม่มี Mask
        )
        
        if results and results[0].boxes:
            # หาตัวที่ Conf สูงสุด
            best_idx = torch.argmax(results[0].boxes.conf).item()
            
            box = results[0].boxes[best_idx]
            
            # 3. Handle Mask: กรณี Detection จะไม่มี Mask (เป็น None)
            mask = None
            if hasattr(results[0], 'masks') and results[0].masks is not None:
                # เผื่ออนาคตกลับมาใช้ model seg ก็ยังทำงานได้
                mask = results[0].masks.data[best_idx]
                
            return box, mask
            
        return None, None
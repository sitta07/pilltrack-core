import cv2
import os
import time

class CameraManager:
    def __init__(self):
        self.cap = None
        self.last_zoom_val = -1 
        self.idx = 0

    def start(self, idx=0):
        self.idx = idx
        if self.cap: self.cap.release()
        
        backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(idx, backend)
        
        # บังคับ Full HD
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if self.cap.isOpened():
            print(f"✅ Camera {idx} Started (1920x1080)")
            # สั่ง Reset Zoom ไปที่ -100 ทันทีที่เปิด (Wide สุด)
            try:
                self.cap.set(cv2.CAP_PROP_ZOOM, -100)
                # print("🔄 Zoom Init: -100")
            except:
                pass
            
            time.sleep(0.5)
            self.trigger_autofocus()
            return True
        return False

    def set_zoom(self, value):
        """
        Hardware Zoom
        value: 1.0 (Wide) -> 4.0 (Zoom)
        """
        if not self.cap: return

        # เช็คกันคำสั่งรัว
        if abs(value - self.last_zoom_val) < 0.05: return
        self.last_zoom_val = value

        # 🔥 FIX FORMULA: เลื่อนสเกลให้เริ่มที่ -100
        driver_value = int((value * 100) - 200) 
        
        try:
            self.cap.set(cv2.CAP_PROP_ZOOM, driver_value)
            # print(f"Set Hardware Zoom: {driver_value}")
        except Exception as e:
            print(f"Zoom Error: {e}")

    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def trigger_autofocus(self):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0) # Off
            time.sleep(0.1)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) # On

    def stop(self):
        if self.cap: self.cap.release()
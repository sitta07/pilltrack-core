import cv2
import os
import time

class CameraManager:
    def __init__(self):
        self.cap = None
        
    def initialize_zoom(self, value=50):
        if self.cap and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                self.cap.set(cv2.CAP_PROP_ZOOM, value)
                current = self.cap.get(cv2.CAP_PROP_ZOOM)
                print(f"🔍 Hardware zoom set to {current}")
            except:
                print("⚠️ Zoom control failed")


    def start(self, idx=0):
        if self.cap: self.cap.release()
        
        # ใช้ V4L2 บน Linux เพื่อความเสถียร
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        if self.cap.isOpened():
            # 🛑 สั่ง Wide สุดครั้งเดียวจบ (Hardware)
            try:
                self.initialize_zoom(50)  # Wide สุด

                print(f"✅ Camera {idx}: Hardware Zoom locked at Wide.")
            except: pass
            return True
        return False


    def get_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            return frame if ret else None
        return None

    def stop(self):
        if self.cap: self.cap.release()
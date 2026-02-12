import sys
import glob
import cv2
from PyQt6.QtWidgets import QApplication
from core.detector import ObjectDetector
from core.processor import ImageProcessor
from ui.station_window import StationWindow
from config import settings

def get_real_cameras():
    """กรองเอาเฉพาะกล้องจริง ไม่เอา Metadata"""
    devs = glob.glob('/dev/video*')
    valid = []
    for d in sorted(devs):
        idx = int(d.replace('/dev/video', ''))
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret: valid.append(idx)
            cap.release()
    
    # กรอง Ghost Device (Linux มักโชว์ video0 กับ video1 เป็นตัวเดียวกัน)
    unique = []
    if valid:
        unique.append(valid[0])
        for i in range(1, len(valid)):
            if valid[i] > valid[i-1] + 1:
                unique.append(valid[i])
    return unique
from PyQt6.QtGui import QGuiApplication

def main():
    app = QApplication(sys.argv)
    
    # ดึงข้อมูลหน้าจอทั้งหมดที่ระบบตรวจเจอ
    screens = QGuiApplication.screens()
    
    detector = ObjectDetector()
    detector.load_model('models/yolov8n-seg.pt')
    processor = ImageProcessor()

    active_cams = get_real_cameras()
    stations = []

    for i, cam_idx in enumerate(active_cams):
        # ป้องกันกรณีจำนวนกล้องมากกว่าจำนวนหน้าจอ
        screen_index = i if i < len(screens) else 0
        target_screen = screens[screen_index]
        screen_geometry = target_screen.geometry()

        win = StationWindow(i, cam_idx, detector, processor)
        
        # ย้ายไปที่หน้าจอนั้นๆ โดยใช้พิกัดจาก System Geometry
        win.move(screen_geometry.left(), screen_geometry.top())
        
        # บังคับขนาดให้เต็มหน้าจอตาม Geometry ของจอนั้นๆ
        win.resize(screen_geometry.width(), screen_geometry.height())
        
        # ถ้าต้องการให้คลุม Taskbar ด้วย
        win.showFullScreen() 
        
        stations.append(win)

    print(f"🚀 System Ready. Deployed on {len(stations)} Screen(s).")
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
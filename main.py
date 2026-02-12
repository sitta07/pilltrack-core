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

def main():
    app = QApplication(sys.argv)
    
    # 1. โหลดทรัพยากรกลาง (Shared AI)
    detector = ObjectDetector()
    # พี่ใช้ RTX 5060 Ti แนะนำใช้ -seg เพื่อตัดพื้นหลังโหดๆ
    detector.load_model('models/yolov8n-seg.pt')
    processor = ImageProcessor()

    # 2. ค้นหากล้อง
    active_cams = get_real_cameras()
    print(f"✅ Found Cameras: {active_cams}")

    stations = []
    # 3. สร้าง Station แยกตามจอ
    for i, cam_idx in enumerate(active_cams):
        win = StationWindow(i, cam_idx, detector, processor)
        
        # ย้ายตำแหน่งไปตามจอ (0, 1920, 3840...)
        x_pos = i * settings.MONITOR_WIDTH
        win.move(x_pos, 0)
        
        # 🔥 สั่ง Full Screen ทันที
        win.showFullScreen() 
        stations.append(win)

    print(f"🚀 PillTrack System is Ready on {len(stations)} Monitor(s).")
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
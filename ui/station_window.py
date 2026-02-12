import cv2
import numpy as np
from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QHBoxLayout
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

class StationWindow(QMainWindow):
    def __init__(self, station_id, camera_idx, detector, processor):
        super().__init__()
        self.station_id = station_id
        self.detector = detector
        self.processor = processor
        self.camera_idx = camera_idx
        
        # โหมดเริ่มต้น: "BOX" (กว้าง) | "PILL" (ซูม)
        self.current_mode = "BOX" 
        
        from core.camera import CameraManager
        self.cam_mgr = CameraManager()
        self.cam_mgr.start(camera_idx)

        self.init_ui()
        self.apply_mode_settings() # ตั้งค่าเริ่มต้น

        # Loop ประมวลผลภาพ
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(33) 

    def init_ui(self):
        """Interface สำหรับ Kiosk Mode"""
        self.setStyleSheet("background-color: #0a0a0a; color: #eee;")
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- ฝั่งซ้าย: Video Feed ---
        self.video_label = QLabel("กำลังเรียกภาพ...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #333; background-color: black;")
        main_layout.addWidget(self.video_label, stretch=7)

        # --- ฝั่งขวา: Info Panel ---
        right_panel = QVBoxLayout()
        
        header = QLabel(f"STATION {self.station_id + 1}")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #00ff00;")
        right_panel.addWidget(header)

        # ช่องโชว์รูปยาที่ Crop ได้
        self.pill_display = QLabel("No Object")
        self.pill_display.setFixedSize(300, 300)
        self.pill_display.setStyleSheet("border: 2px dashed #444; background-color: #000;")
        self.pill_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.pill_display)

        # สถานะและปุ่มกด
        self.info_label = QLabel("โหมด: กล่องยา (กว้าง)")
        self.info_label.setFont(QFont("Arial", 14))
        self.info_label.setStyleSheet("background-color: #1a1a1a; padding: 10px; border-radius: 5px;")
        right_panel.addWidget(self.info_label)

        hint = QLabel("กด [1] กล่องยา | [2] เม็ดยา\nกด [Q] ปิด | [Esc] ย่อจอ")
        hint.setStyleSheet("color: #888;")
        right_panel.addWidget(hint)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=3)

    def apply_mode_settings(self):
        """ส่งคำสั่งไป Hardware เฉพาะตอนเปลี่ยนโหมดเท่านั้น (กันภาพสั่น)"""
        if self.current_mode == "BOX":
            self.cam_mgr.set_zoom(-100) # Wide สุด
            self.info_label.setText("โหมด: กล่องยา (Wide)")
        else:
            self.cam_mgr.set_zoom(100) # Zoom เข้าหาถาด
            self.info_label.setText("โหมด: เม็ดยา (Zoom)")
        
        # สะกิดโฟกัสใหม่ 1 ที
        self.cam_mgr.trigger_autofocus()

    def keyPressEvent(self, event):
        """Keyboard Shortcuts"""
        if event.key() == Qt.Key.Key_1:
            self.current_mode = "BOX"
            self.apply_mode_settings()
        elif event.key() == Qt.Key.Key_2:
            self.current_mode = "PILL"
            self.apply_mode_settings()
        elif event.key() == Qt.Key.Key_Escape:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key.Key_Q:
            self.close()

    def update_logic(self):
        frame = self.cam_mgr.get_frame()
        if frame is None: return

        # AI Predict (Unpack Box & Mask)
        box, mask = self.detector.predict(frame, conf=0.1) # Confidence 0.1

        if box is not None:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ตัดพื้นหลังเนียนๆ ด้วย AI Mask (ถ้ามี)
            if mask is not None:
                pill_cutout = self.processor.cutout_by_mask(frame, box, mask)
                if pill_cutout is not None:
                    self.display_pill(pill_cutout, conf)
        
        self.display_main_video(frame)

    def display_main_video(self, frame):
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        qt_img = QImage(rgb_img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def display_pill(self, pill_img, conf):
        rgb_pill = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_pill.shape
        qt_pill = QImage(rgb_pill.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.pill_display.setPixmap(QPixmap.fromImage(qt_pill).scaled(
            self.pill_display.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        self.cam_mgr.stop()
        event.accept()
import cv2
import numpy as np
import time
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout,
    QWidget, QHBoxLayout, QPushButton
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap, QFont

# Import Core Modules
from core.camera import CameraManager
# 🔥 UPDATE: เปลี่ยนชื่อ Import เป็น SmartClassifier
from core.classifier import SmartClassifier 

class StationWindow(QMainWindow):

    def __init__(self, station_id, camera_idx, detector, processor):
        super().__init__()

        self.station_id = station_id
        self.detector = detector
        self.processor = processor
        self.camera_idx = camera_idx

        # 1. Setup Camera
        self.cam_mgr = CameraManager()
        self.cam_mgr.start(camera_idx)

        # 2. Setup Classifier (Smart Version)
        print("⏳ Loading Smart Classifier...")
        # 🔥 UPDATE: เรียกใช้ SmartClassifier (มันจะโหลด 2 โมเดลรอข้างในเอง)
        self.classifier = SmartClassifier()
        print("✅ Classifier Ready!")

        # ====== MODE & STATE ======
        self.mode = "PILL"   # PILL or BOX
        self.start_time = time.time()
        self.focus_phase = True  # 1 วิแรก focus boost

        self.init_ui()

        # Timer 30 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(33)

    # ================= UI SETUP ================= #

    def init_ui(self):
        self.setWindowTitle("Pill Inspection Station")
        self.setStyleSheet("background-color: black; color: white;")

        self.showFullScreen()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ===== LEFT: VIDEO =====
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.video_label, stretch=8)

        # ===== RIGHT PANEL =====
        right_panel = QVBoxLayout()

        self.header = QLabel()
        self.header.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        right_panel.addWidget(self.header)

        self.pill_display = QLabel("No Object")
        self.pill_display.setFixedSize(300, 300)
        self.pill_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pill_display.setStyleSheet(
            "border: 2px dashed #444; background:black;"
        )
        right_panel.addWidget(self.pill_display)

        # Result Label
        self.result_label = QLabel("") 
        self.result_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.result_label)

        self.info_label = QLabel("กำลังเริ่มระบบ...")
        self.info_label.setFont(QFont("Arial", 14))
        right_panel.addWidget(self.info_label)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=2)

    # ================= KEY EVENTS ================= #

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()

        elif event.key() == Qt.Key.Key_M:
            self.toggle_mode()

        elif event.key() == Qt.Key.Key_Escape:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

    def toggle_mode(self):
        previous_mode = self.mode
        self.mode = "BOX" if self.mode == "PILL" else "PILL"

        # 🔥 UPDATE: สั่งสลับสมอง (Model) ผ่าน SmartClassifier
        self.classifier.switch_mode(self.mode)

        if self.mode == "PILL":
            # Hardware Zoom & Focus Logic
            self.cam_mgr.initialize_zoom(80)  
            self.start_time = time.time()
            self.focus_phase = True
        else:
            # Hardware Zoom
            self.cam_mgr.initialize_zoom(50)
            self.focus_phase = False # Box Mode ไม่ต้องใช้ Focus Phase

        self.info_label.setText(f"Switch Mode: {previous_mode} -> {self.mode}")
        self.update_header()

    # ================= DIGITAL ZOOM ================= #

    def digital_zoom(self, frame, zoom_factor):
        if zoom_factor <= 1.0:
            return frame

        h, w = frame.shape[:2]
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2

        cropped = frame[y1:y1+new_h, x1:x1+new_w]
        return cv2.resize(cropped, (w, h))

    # ================= MAIN LOOP ================= #

    def update_logic(self):
        frame = self.cam_mgr.get_frame()
        if frame is None:
            return

        elapsed = time.time() - self.start_time

        # ===== Zoom Logic =====
        if self.focus_phase:
            frame = self.digital_zoom(frame, 2.0)
            if elapsed > 1.0:
                self.focus_phase = False
                self.info_label.setText("พร้อมทำงาน")
        else:
            if self.mode == "PILL":
                frame = self.digital_zoom(frame, 2.0)
            else:
                frame = frame 

        # ===== AI Detection =====
        box, mask = self.detector.predict(frame, conf=0.25)

        if box is not None:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if mask is not None:
                pill_cutout = self.processor.cutout_by_mask(frame, box, mask)
                
                if pill_cutout is not None:
                    h, w = pill_cutout.shape[:2]
                    if h > 10 and w > 10:
                        # 🔥 Predict (ข้างในมันเลือก Model ให้เองตาม Mode)
                        name, conf = self.classifier.predict(pill_cutout)
                        self.display_pill(pill_cutout, name, conf)
                    else:
                        self.info_label.setText("วัตถุเล็กเกินไป")

        self.display_main_video(frame)
        self.update_header()

    # ================= DISPLAY ================= #

    def display_main_video(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        scaled = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        self.video_label.setPixmap(scaled)

    def display_pill(self, pill_img, name, conf):
        rgb = cv2.cvtColor(pill_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape

        qt_img = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)

        scaled = pixmap.scaled(
            300, 300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.pill_display.setPixmap(scaled)

        percentage = conf * 100
        color = "#00FF00" if percentage > 70 else "#FFA500"
        
        self.result_label.setText(f"{name}\n{percentage:.1f}%")
        self.result_label.setStyleSheet(f"color: {color};")
        self.info_label.setText(f"ตรวจพบวัตถุ")

    def update_header(self):
        self.header.setText(
            f"STATION {self.station_id + 1} | MODE: {self.mode}"
        )

    def closeEvent(self, event):
        self.cam_mgr.stop()
        self.timer.stop()
        event.accept()
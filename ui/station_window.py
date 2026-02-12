# ui/station_window.py
import os
import time
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage, QFont

from .inference import PillPredictor  # safe predict
from core.camera import CameraManager  # assumes you have CameraManager

# YOLO detector class (replace with your actual YOLO wrapper)
from .yolov8_wrapper import YOLODetector  # คุณต้องมี wrapper ของ YOLO

class StationWindow(QMainWindow):
    def __init__(self, station_id, camera_idx, detector, processor, device="cpu"):
        super().__init__()
        self.station_id = station_id
        self.camera_idx = camera_idx
        self.detector = detector
        self.processor = processor
        self.device = device

        # ===== Camera =====
        self.cam_mgr = CameraManager()
        self.cam_mgr.start(camera_idx)

        # ===== YOLO =====
        self.detector = YOLODetector("yolov8n-seg.pt", device=device)
        self.processor = self.detector  # ใช้ cutout_by_mask จาก YOLO wrapper

        # ===== MODE =====
        self.mode = "PILL"  # PILL / BOX
        self.start_time = time.time()
        self.focus_phase = True

        # ===== PillPredictor =====
        self.pill_predictor = PillPredictor(
            model_path="models/best_model_pill.pth",
            class_map_path="models/class_mapping_pill.json",
            device=device
        )
        self.box_predictor = PillPredictor(
            model_path="models/best_model_box.pth",
            class_map_path="models/class_mapping_box.json",
            device=device
        )

        self.init_ui()

        # ===== Timer =====
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(33)

    # ================= UI =================
    def init_ui(self):
        self.setWindowTitle("Pill Inspection Station")
        self.setStyleSheet("background-color: black; color: white;")
        self.showFullScreen()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.video_label, stretch=8)

        # Right panel
        right_panel = QVBoxLayout()

        # Header
        self.header = QLabel(f"STATION {self.station_id+1} | MODE: {self.mode}")
        self.header.setFont(QFont("Arial", 22, QFont.Weight.Bold))
        right_panel.addWidget(self.header)

        # Crop preview
        self.pill_display = QLabel("No Object")
        self.pill_display.setFixedSize(300, 300)
        self.pill_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pill_display.setStyleSheet("border: 2px dashed #444; background:black;")
        right_panel.addWidget(self.pill_display)

        # Info label
        self.info_label = QLabel("Initializing...")
        self.info_label.setFont(QFont("Arial", 14))
        right_panel.addWidget(self.info_label)

        # Toggle mode button (touch screen)
        self.toggle_button = QPushButton("Toggle Mode")
        self.toggle_button.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.toggle_button.setStyleSheet(
            "background-color:#444; color:white; padding:10px; border-radius:10px;"
        )
        self.toggle_button.clicked.connect(self.toggle_mode)
        right_panel.addWidget(self.toggle_button)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=2)

    # ================= KEY =================
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Q:
            self.close()
        elif event.key() == Qt.Key.Key_M:
            self.toggle_mode()
        elif event.key() == Qt.Key.Key_Escape:
            self.showNormal()

    def toggle_mode(self):
        self.mode = "BOX" if self.mode == "PILL" else "PILL"
        self.header.setText(f"STATION {self.station_id+1} | MODE: {self.mode}")
        self.info_label.setText(f"Switched to {self.mode}")

        # Optional: camera zoom
        if self.mode == "PILL":
            self.cam_mgr.initialize_zoom(85)
            QTimer.singleShot(1000, lambda: self.cam_mgr.initialize_zoom(65))
        else:
            self.cam_mgr.initialize_zoom(50)

    # ================= DIGITAL ZOOM =================
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

    # ================= LOOP =================
    def update_logic(self):
        frame = self.cam_mgr.get_frame()
        if frame is None:
            return

        elapsed = time.time() - self.start_time

        # Focus phase 1 sec
        if self.focus_phase:
            frame = self.digital_zoom(frame, 2.0)
            if elapsed > 1.0:
                self.focus_phase = False
                self.info_label.setText("Ready")
        else:
            if self.mode == "PILL":
                frame = self.digital_zoom(frame, 2)
            # BOX = no zoom

        # ===== YOLO detect + crop =====
        box, mask = self.detector.predict(frame, conf=0.1)
        pill_cutout = None
        if box is not None:
            pill_cutout = self.processor.cutout_by_mask(frame, box, mask)
            if pill_cutout is not None:
                # ===== PREDICT CLASS =====
                predictor = self.pill_predictor if self.mode=="PILL" else self.box_predictor
                cls_name, conf = predictor.predict(pill_cutout)
                self.display_pill(pill_cutout, cls_name, conf)

        self.display_main_video(frame)

    # ================= DISPLAY =================
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

    def display_pill(self, pill_img, class_name, conf):
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
        self.info_label.setText(f"Found: {class_name} | Conf: {conf:.2f}")

    # ================= CLOSE =================
    def closeEvent(self, event):
        self.cam_mgr.stop()
        event.accept()

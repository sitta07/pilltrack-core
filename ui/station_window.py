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
        
        from core.camera import CameraManager
        self.cam_mgr = CameraManager()
        self.cam_mgr.start(camera_idx)

        self.init_ui()

        # Timer สำหรับ Loop 30 FPS
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_logic)
        self.timer.start(33) 

    def init_ui(self):
        """เน้น UI ที่สะอาดสำหรับใช้งานจริงในรพ."""
        self.setStyleSheet("background-color: #0a0a0a; color: #ffffff;")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- ฝั่งซ้าย: จอหลัก (ภาพสด Wide) ---
        self.video_label = QLabel("Camera Feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("border: 1px solid #333; background-color: black;")
        main_layout.addWidget(self.video_label, stretch=7)

        # --- ฝั่งขวา: ผลการวิเคราะห์ ---
        right_panel = QVBoxLayout()
        
        header = QLabel(f"STATION {self.station_id + 1}")
        header.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        header.setStyleSheet("color: #00ff00; margin-bottom: 10px;")
        right_panel.addWidget(header)

        # ภาพ Crop เม็ดยา (ตัด BG)
        self.pill_display = QLabel("No Object")
        self.pill_display.setFixedSize(300, 300)
        self.pill_display.setStyleSheet("border: 2px dashed #444; background-color: #000; border-radius: 10px;")
        self.pill_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_panel.addWidget(self.pill_display)

        self.info_label = QLabel("สถานะ: รอยาวางบนถาด...")
        self.info_label.setFont(QFont("Arial", 14))
        self.info_label.setStyleSheet("background-color: #1a1a1a; padding: 15px; border-radius: 10px;")
        right_panel.addWidget(self.info_label)

        # คำแนะนำสั้นๆ
        footer = QLabel("กด [Esc] ย่อ/ขยายจอ | [Q] ปิด")
        footer.setStyleSheet("color: #666; font-size: 10pt;")
        right_panel.addWidget(footer)

        right_panel.addStretch()
        main_layout.addLayout(right_panel, stretch=3)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        elif event.key() == Qt.Key.Key_Q:
            self.close()

    def update_logic(self):
        """ดึงภาพมาประมวลผล (ไม่มีการสั่งซูมใน Loop นี้แล้ว)"""
        frame = self.cam_mgr.get_frame()
        if frame is None: return

        # 🔥 ส่งภาพดิบเข้า AI เลย (ไม่ผ่าน apply_filters ที่แอบมี digital zoom)
        box, mask = self.detector.predict(frame, conf=0.1)

        if box is not None:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            
            # วาดกรอบแค่ให้รู้ว่าเจอ (เขียว)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ตัด Background (AI Seg)
            if mask is not None:
                pill_cutout = self.processor.cutout_by_mask(frame, box, mask)
                if pill_cutout is not None:
                    self.display_pill(pill_cutout, conf)
        
        # แสดงภาพสด (Wide)
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
        self.info_label.setText(f"<b>ตรวจพบวัตถุ</b><br>ความมั่นใจ: {conf:.2f}")

    def closeEvent(self, event):
        self.cam_mgr.stop()
        event.accept()
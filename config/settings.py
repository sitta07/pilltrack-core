import numpy as np

# Display Settings
MONITOR_WIDTH = 1920  # ความกว้างจอภาพ (สำคัญมากสำหรับการย้ายหน้าต่าง)
MONITOR_HEIGHT = 1080

# Camera Settings
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# AI & Processing
CONFIDENCE_THRESHOLD = 0.1
THUMB_SIZE = (150, 150) # ขนาดรูปเล็กมุมขวาบน (กว้าง, สูง)

# Green Screen (HSV) - ปรับจูนตามแสงจริงหน้างาน
LOWER_GREEN = np.array([35, 40, 40])
UPPER_GREEN = np.array([85, 255, 255])
import os
import yaml
import cv2
import numpy as np

# ==========================================
# ⚙️ CONFIG & SYSTEM UTILS
# ==========================================
def load_config(config_path="config/settings.yaml"):
    """อ่านไฟล์ Config"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(full_config_path):
        full_config_path = os.path.abspath(config_path)
        if not os.path.exists(full_config_path):
             raise FileNotFoundError(f"❌ Config file not found at: {full_config_path}")
        base_dir = os.path.dirname(full_config_path)
        
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['system'] = config.get('system', {})
    config['system']['base_dir'] = base_dir
    return config

def find_working_camera():
    """วนหา Camera Index"""
    print("🔍 Searching for available camera...")
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.release()
                print(f"✅ Found working camera at Index: {index}")
                return index
            cap.release()
    print("❌ No physical camera found. Using dummy mode.")
    return None

# ==========================================
# 🖼️ IMAGE PROCESSING UTILS
# ==========================================
def get_auto_hsv_bounds(frame, sample_size=30):
    if frame is None or frame.shape[0] < sample_size or frame.shape[1] < sample_size:
        return np.array([0,0,0]), np.array([180,255,255])
    h, w, _ = frame.shape
    tl = frame[0:sample_size, 0:sample_size]
    tr = frame[0:sample_size, w-sample_size:w]
    bl = frame[h-sample_size:h, 0:sample_size]
    br = frame[h-sample_size:h, w-sample_size:w]
    samples = np.vstack((tl, tr, bl, br))
    hsv_samples = cv2.cvtColor(samples, cv2.COLOR_BGR2HSV)
    mean = np.mean(hsv_samples, axis=(0, 1))
    lower = np.clip(mean - np.array([20, 50, 50]), 0, 255).astype(np.uint8)
    upper = np.clip(mean + np.array([20, 50, 50]), 0, 255).astype(np.uint8)
    return lower, upper

def remove_green_bg_auto(image):
    """ตัดพื้นหลังเขียว/ดำอัตโนมัติ (สำหรับโหมดปกติ)"""
    if image is None or image.size == 0: return image
    lower, upper = get_auto_hsv_bounds(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

# 🔥🔥🔥 ฟังก์ชันพระเอกของเรา (Polygon Masking) 🔥🔥🔥
def apply_polygon_mask(image, polygon, crop_offset):
    """
    ใช้ Polygon (จุดพิกัด) ตัดขอบยา รับประกันว่าทรงไม่เบี้ยว 100%
    
    Args:
        image: ภาพ Pill Crop (เม็ดเดียว)
        polygon: จุดพิกัดรอบเม็ดยา (Global Coordinates จากภาพใหญ่)
        crop_offset: (x1, y1) จุดมุมซ้ายบนของ Pill Crop
    """
    if image is None or polygon is None or len(polygon) == 0: return image
    
    h, w = image.shape[:2]
    crop_x, crop_y = crop_offset
    
    # 1. สร้างหน้ากากสีดำขนาดเท่าภาพ Crop
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # 2. แปลงพิกัดจาก Global (ทั้งภาพ) -> Local (เฉพาะใน Crop)
    # สูตร: จุดใน crop = จุดจริง - จุดเริ่มต้น crop
    # ต้อง copy ออกมาเพื่อไม่ให้กระทบค่าเดิม
    local_polygon = polygon.copy()
    local_polygon[:, 0] -= crop_x
    local_polygon[:, 1] -= crop_y
    
    # 3. วาดรูปทรงยาลงบนหน้ากาก (Filled Polygon = สีขาว 255)
    # ต้องแปลงเป็น int32 ก่อนวาดด้วย opencv
    points = local_polygon.astype(np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    # 4. ตัดภาพ (พื้นหลังจะเป็นสีดำสนิท)
    result = cv2.bitwise_and(image, image, mask=mask)
    
    return result
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
        # Fallback case
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
    """วนหา Camera Index (0-9)"""
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
    """ตัดพื้นหลังเขียว/ดำอัตโนมัติ"""
    if image is None or image.size == 0: return image
    lower, upper = get_auto_hsv_bounds(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask_inv = cv2.bitwise_not(mask)
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result

def apply_yolo_mask(image, mask_data):
    """
    ใช้ Mask จาก YOLO Segmentation ตัดพื้นหลัง
    image: ภาพ Pill Crop
    mask_data: ข้อมูล Mask (numpy array)
    """
    if image is None or mask_data is None: return image
    h, w = image.shape[:2]
    
    # Resize mask ให้เท่ากับภาพ (เผื่อขนาดไม่ตรง)
    mask_resized = cv2.resize(mask_data, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Convert to binary mask (0 or 255)
    mask_uint8 = (mask_resized * 255).astype(np.uint8)
    
    # Cut background
    result = cv2.bitwise_and(image, image, mask=mask_uint8)
    return result
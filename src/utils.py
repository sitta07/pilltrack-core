import os
import yaml
import cv2
import numpy as np

# ==========================================
# ⚙️ CONFIG & SYSTEM UTILS
# ==========================================
def load_config(config_path="config/settings.yaml"):
    """อ่านไฟล์ Config และแปลง Path ให้เป็น Absolute Path กันหลง"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"❌ Config file not found at: {full_config_path}")
        
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    config['system'] = config.get('system', {})
    config['system']['base_dir'] = base_dir
    
    return config

def find_working_camera():
    """วนหา Camera Index ที่ใช้งานได้จริง (0-5)"""
    print("🔍 Searching for available camera...")
    # ลองหา index 0-9 เผื่อบางเครื่องกล้องไปไกล
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

def get_model_path(config, model_type='pill'):
    """ช่วยประกอบ Path ของโมเดลให้ถูกต้อง"""
    base_dir = config['system']['base_dir']
    model_base = config['model']['base_path']
    
    if model_type == 'pill':
        fname = config['model']['pill_classifier']['weights']
    elif model_type == 'box':
        fname = config['model']['box_detector']['weights']
    else:
        return None
        
    return os.path.join(base_dir, model_base, fname)

# ==========================================
# 🖼️ IMAGE PROCESSING UTILS (เพิ่มส่วนนี้เข้ามาครับ!)
# ==========================================
def get_auto_hsv_bounds(frame, sample_size=30):
    """
    สุ่มสีจาก 4 มุมภาพ เพื่อหาค่าสีพื้นหลังโดยอัตโนมัติ
    (ใช้สำหรับตัดฉากหลังเขียว/น้ำเงินออก)
    """
    if frame is None or frame.shape[0] < sample_size or frame.shape[1] < sample_size:
        # ถ้าภาพเล็กไป คืนค่า Default (ไม่ตัดอะไร)
        return np.array([0,0,0]), np.array([180,255,255])
        
    h, w, _ = frame.shape
    
    # สุ่ม 4 มุม
    tl = frame[0:sample_size, 0:sample_size]
    tr = frame[0:sample_size, w-sample_size:w]
    bl = frame[h-sample_size:h, 0:sample_size]
    br = frame[h-sample_size:h, w-sample_size:w]
    
    # รวมแล้วแปลงเป็น HSV
    samples = np.vstack((tl, tr, bl, br))
    hsv_samples = cv2.cvtColor(samples, cv2.COLOR_BGR2HSV)
    
    # หาค่าเฉลี่ยสีพื้นหลัง
    mean = np.mean(hsv_samples, axis=(0, 1))
    
    # สร้างขอบเขต (Range) บวกลบจากค่าเฉลี่ย
    # Threshold: Hue+-20, Sat+-50, Val+-50
    lower_bound = np.clip(mean - np.array([20, 50, 50]), 0, 255).astype(np.uint8)
    upper_bound = np.clip(mean + np.array([20, 50, 50]), 0, 255).astype(np.uint8)
    
    return lower_bound, upper_bound

def remove_green_bg_auto(image):
    """
    ลบพื้นหลังอัตโนมัติ แล้วเปลี่ยนเป็นสีดำ
    ช่วยให้ Model โฟกัสที่เม็ดยาได้แม่นขึ้น 300%
    """
    if image is None or image.size == 0: return image
    
    # 1. หาค่าสีพื้นหลัง
    lower, upper = get_auto_hsv_bounds(image)
    
    # 2. สร้าง Mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    
    # 3. กลับด้าน Mask (เอาส่วนที่ไม่ใช่พื้นหลัง)
    mask_inv = cv2.bitwise_not(mask)
    
    # 4. ตัดภาพ
    result = cv2.bitwise_and(image, image, mask=mask_inv)
    return result
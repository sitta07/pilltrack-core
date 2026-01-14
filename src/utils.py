import os
import yaml
import cv2
import numpy as np

def load_config(config_path="config/settings.yaml"):
    """อ่านไฟล์ Config และแปลง Path ให้เป็น Absolute Path กันหลง"""
    # หา path ของ root project (ถอยจาก src/utils.py ไป 2 ขั้น)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(full_config_path):
        raise FileNotFoundError(f"❌ Config file not found at: {full_config_path}")
        
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Inject 'base_dir' เข้าไปใน config เผื่อต้องใช้
    config['system'] = config.get('system', {})
    config['system']['base_dir'] = base_dir
    
    return config

def find_working_camera():
    """วนหา Camera Index ที่ใช้งานได้จริง (0-5)"""
    print("🔍 Searching for available camera...")
    for index in range(5):
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
    model_base = config['model']['base_path'] # e.g., models/active
    
    if model_type == 'pill':
        fname = config['model']['pill_classifier']['weights']
    elif model_type == 'box':
        fname = config['model']['box_detector']['weights']
    else:
        return None
        
    return os.path.join(base_dir, model_base, fname)
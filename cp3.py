import cv2
import numpy as np
import torch
from src.utils import load_config
from src.models import AIEngine

def check_step_3():
    print("🚦 CHECKPOINT 3: Testing AI Brain (Models)...")
    print("="*60)
    
    # 1. Load Config
    cfg = load_config()
    
    # 2. Init AI Engine
    print("🧠 Initializing AI Engine...")
    try:
        engine = AIEngine(cfg)
    except Exception as e:
        print(f"❌ Critical Error initializing engine: {e}")
        return

    # 3. Test Pill Classification (Dummy Input)
    print("\n💊 Testing Pill Inference (with dummy image)...")
    if engine.pill_model:
        # Create a fake green pill image
        dummy_pill = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(dummy_pill, (150, 150), 100, (255, 255, 255), -1) # White circle
        
        name, conf = engine.predict_pill(dummy_pill)
        print(f"   👉 Prediction Result: {name} (Conf: {conf:.4f})")
        print("   ✅ Pill Inference pipeline is working!")
    else:
        print("   ❌ Pill model is NOT loaded.")

    # 4. Test Box Detection (Dummy Input)
    print("\n📦 Testing Box Detection...")
    if engine.box_model:
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        boxes = engine.predict_box(dummy_frame)
        print(f"   👉 Detection ran successfully (Found {len(boxes)} boxes in black image)")
        print("   ✅ Box Inference pipeline is working!")
    else:
        print("   ⚠️ Box model is NOT loaded (Check if .onnx exists)")

    print("="*60)
    if engine.pill_model:
        print("🎉 PASSED! สมองกลพร้อมทำงาน (ไปต่อ Step สุดท้าย: Web UI)")
    else:
        print("🛑 FAILED! กรุณาเช็คไฟล์ .pth ใน models/active/")

if __name__ == "__main__":
    check_step_3()
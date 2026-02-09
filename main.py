import os
import sys

# ============================================================
# 🛠️ BLACKWELL ENGINE PATCH (RTX 50 SERIES FIX)
# ============================================================
# บังคับให้โหลด CUDA Module แบบ Lazy เพื่อป้องกัน Core Dumped ตอน Start
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# ตั้งค่า Flag สำหรับ ONNX Runtime ให้ทำงานร่วมกับ CUDA 12.8 ได้เสถียรขึ้น
os.environ["ORT_CUDA_FLAGS"] = "1"

import time
import signal
from src.utils import load_config, find_working_camera
from src.camera import WebcamStream
from src.models import AIEngine
from src.web_server import start_server

def main():
    print("\n" + "="*60)
    print("🚀 PILLTRACK PRO-CORE: MEDICAL AI STATION ACTIVE")
    print("="*60)

    # 1. 📂 Load Configuration
    try:
        cfg = load_config()
        print(f"✅ [1/4] Config Loaded Successfully")
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return

    # 2. 📷 Initialize Camera (The Eyes)
    cam_idx = find_working_camera()
    if cam_idx is None:
        print("⚠️ Warning: No physical camera found.")
    
    print(f"📷 [2/4] Initializing Camera Stream (Index: {cam_idx})...")
    # ใช้ WebcamStream แบบ Multi-threaded เพื่อความลื่นไหล
    camera = WebcamStream(
        src=cam_idx, 
        width=cfg['camera'].get('width', 1280), 
        height=cfg['camera'].get('height', 720)
    ).start()
    
    # ให้เวลากล้อง Warm-up และปรับแสงอัตโนมัติ
    time.sleep(2.0) 

    # 3. 🧠 Initialize AI Engine (The Brain)
    print("🧠 [3/4] Warming up AI Engine on GPU (RTX 5060 Ti)...")
    try:
        # โหลดทั้ง YOLO และ Classifier เข้าสู่ Memory GPU
        engine = AIEngine(cfg)
        print(f"✅ AI Engine Ready: Blackwell Optimized")
    except Exception as e:
        print(f"❌ AI Engine Initialization Failed: {e}")
        if 'camera' in locals(): camera.stop()
        return

    # 4. 🌍 Start Web Server & AI Pipeline (The Service)
    print("="*60)
    print("🌍 [4/4] Starting Web Interface: http://localhost:5000")
    try:
        # ฟังก์ชันนี้จะบล็อกการทำงาน (Blocking) เพื่อรัน Flask และ AI Worker
        start_server(camera, engine, cfg)
    except KeyboardInterrupt:
        print("\n🛑 User Interrupted: Stopping System...")
    except Exception as e:
        print(f"🔥 Unexpected Runtime Error: {e}")
    finally:
        # 🧹 Graceful Shutdown: เคลียร์ทรัพยากรทุกอย่าง
        print("\n" + "="*60)
        print("🧹 Cleaning up system resources...")
        if 'camera' in locals():
            camera.stop()
        print("✅ System Offline.")
        print("👋 System Shutdown Complete.")

if __name__ == "__main__":
    # ตรวจสอบว่ามีโฟลเดอร์ __pycache__ ค้างอยู่ไหม ถ้ามีให้เคลียร์ก่อนรัน
    os.system('find . -name "__pycache__" -type d -exec rm -rf {} + > /dev/null 2>&1')
    main()
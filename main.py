import sys
import time
from src.utils import load_config, find_working_camera
from src.camera import WebcamStream
from src.models import AIEngine
from src.web_server import start_server

def main():
    print("🚀 STARTING PILLTRACK SYSTEM (PRODUCTION)")
    print("="*60)

    # 1. Load Config
    try:
        cfg = load_config()
        print(f"✅ Config Loaded")
    except Exception as e:
        print(f"❌ Config Error: {e}")
        return

    # 2. Init Camera
    cam_idx = find_working_camera()
    if cam_idx is None:
        print("⚠️ No camera found, starting in Dummy Mode")
    
    print("📷 Initializing Camera Stream...")
    camera = WebcamStream(src=cam_idx, 
                         width=cfg['camera']['width'], 
                         height=cfg['camera']['height']).start()
    time.sleep(2.0) # Warm up

    # 3. Init AI Brain
    print("🧠 Initializing AI Engine (this may take a moment)...")
    try:
        engine = AIEngine(cfg)
    except Exception as e:
        print(f"❌ AI Engine Failed: {e}")
        camera.stop()
        return

    # 4. Start UI (Blocking)
    print("="*60)
    try:
        start_server(camera, engine, cfg)
    except KeyboardInterrupt:
        print("\n🛑 Stopping System...")
    finally:
        camera.stop()
        print("👋 System Shutdown Complete.")

if __name__ == "__main__":
    main()
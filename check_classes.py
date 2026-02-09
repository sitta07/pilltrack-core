import json
import os

# Config path ให้ตรงกับที่เราโหลดมา
BASE_DIR = "models"
MODELS = {
    "💊 PILL MODEL": os.path.join(BASE_DIR, "pill", "class_mapping.json"),
    "📦 BOX MODEL":  os.path.join(BASE_DIR, "box", "class_mapping.json")
}

def load_and_print_classes(model_name, json_path):
    print(f"\n{'='*50}")
    print(f"Checking: {model_name}")
    print(f"Path: {json_path}")
    print(f"{'-'*50}")

    if not os.path.exists(json_path):
        print(f"❌ Error: File not found! ({json_path})")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # แปลง Key จาก String เป็น Int แล้วเรียงลำดับ
        sorted_classes = sorted(data.items(), key=lambda x: int(x[0]))
        
        print(f"✅ Found {len(sorted_classes)} classes:\n")
        
        # จัด Format การปริ้นให้สวยงาม (3 คอลัมน์)
        for idx, (class_id, class_name) in enumerate(sorted_classes):
            print(f"  [{class_id:>2}] {class_name:<25}", end="")
            if (idx + 1) % 3 == 0: # ขึ้นบรรทัดใหม่ทุก 3 ตัว
                print()
        print("\n")
        
    except Exception as e:
        print(f"❌ Error reading JSON: {e}")

def main():
    print("🚀 STARTING MODEL INSPECTION...")
    
    for name, path in MODELS.items():
        load_and_print_classes(name, path)

    print(f"{'='*50}")
    print("✅ Inspection Complete.")

if __name__ == "__main__":
    main()

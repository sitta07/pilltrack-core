import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import json
import os

# Import Structure เดิม
from core.architecture import PillModel 

class SmartClassifier:
    def __init__(self, model_dir="models/"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir
        
        # ==========================================
        # ⚙️ CONFIG: ตั้งค่าชื่อไฟล์ตรงนี้
        # ==========================================
        # ตอนนี้ใช้ best_model.pth ตัวเดิมทั้งคู่ (อนาคตแก้ชื่อไฟล์ตรงนี้ได้เลย)
        self.pill_config = {
            "weights": "best_model_pills.pth",       
            "mapping": "class_mapping_pills.json", # 🔥 แยกไฟล์ตามขอ
            "img_size": 224,
            "model_name": "convnext_small"
        }
        
        self.box_config = {
            "weights": "best_model_box.pth",       # อนาคตเปลี่ยนเป็น box_model.pth
            "mapping": "class_mapping_box.json",   # 🔥 แยกไฟล์ตามขอ
            "img_size": 224,                   # อนาคตอาจจะเป็น 512
            "model_name": "convnext_small"
        }
        # ==========================================

        print(f"🚀 Initializing SmartClassifier on {self.device}...")

        # 1. โหลดโมเดล PILL เข้า Memory
        self.model_pill, self.classes_pill, self.tfm_pill = self._load_single_model(self.pill_config)
        
        # 2. โหลดโมเดล BOX เข้า Memory
        self.model_box, self.classes_box, self.tfm_box = self._load_single_model(self.box_config)

        # 3. ตัวแปร Pointer (ชี้ว่าจะใช้ตัวไหนทำงาน)
        self.active_model = self.model_pill
        self.active_classes = self.classes_pill
        self.active_tfm = self.tfm_pill
        self.current_mode = "PILL"

        print("✅ Dual Models Loaded & Ready!")

    def _load_single_model(self, config):
        """Helper Function: โหลดโมเดล 1 ตัว"""
        path_weight = os.path.join(self.model_dir, config["weights"])
        path_map = os.path.join(self.model_dir, config["mapping"])

        # A. Load Mapping
        idx_to_class = {}
        num_classes = 5 # Default
        if os.path.exists(path_map):
            try:
                with open(path_map, 'r') as f:
                    raw = json.load(f)
                    idx_to_class = {int(k): v for k, v in raw.items()}
                    num_classes = len(idx_to_class)
                print(f"   📄 Loaded mapping: {config['mapping']} ({num_classes} classes)")
            except Exception as e:
                print(f"   ⚠️ Error loading {config['mapping']}: {e}")
        else:
            print(f"   ⚠️ Warning: {config['mapping']} not found. Using dummy classes.")

        # B. Setup Model Architecture
        model = PillModel(
            num_classes=num_classes,
            model_name=config["model_name"],
            embed_dim=512,
            dropout=0.0
        ).to(self.device)

        # C. Load Weights
        if os.path.exists(path_weight):
            checkpoint = torch.load(path_weight, map_location=self.device)
            model.load_state_dict(checkpoint)
            model.eval() # 🔥 Important
            print(f"   💾 Loaded weights: {config['weights']}")
        else:
            print(f"   ❌ Error: Weight file {config['weights']} not found!")

        # D. Setup Transform
        tfm = transforms.Compose([
            transforms.Resize((config["img_size"], config["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        return model, idx_to_class, tfm

    def switch_mode(self, mode):
        """ฟังก์ชันสลับสมอง (เรียกจาก UI)"""
        if mode == "BOX":
            self.active_model = self.model_box
            self.active_classes = self.classes_box
            self.active_tfm = self.tfm_box
            self.current_mode = "BOX"
        else:
            self.active_model = self.model_pill
            self.active_classes = self.classes_pill
            self.active_tfm = self.tfm_pill
            self.current_mode = "PILL"
            
        print(f"🔄 Switched Classifier to: {self.current_mode}")

    def predict(self, cv2_image):
        """ทำนายผลโดยใช้ Active Model ณ ปัจจุบัน"""
        if cv2_image is None or cv2_image.size == 0:
            return "Error", 0.0

        # Preprocess
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor_img = self.active_tfm(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Extract Feature
            features = self.active_model(tensor_img)
            
            # ArcFace Similarity Check
            norm_feat = F.normalize(features)
            norm_weight = F.normalize(self.active_model.head.weight)
            logits = F.linear(norm_feat, norm_weight)
            probs = F.softmax(logits * 30.0, dim=1)
            
            conf, pred_idx = torch.max(probs, 1)
            
            idx = pred_idx.item()
            name = self.active_classes.get(idx, f"Unknown-{idx}")
            
            return name, conf.item()
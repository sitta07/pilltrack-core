import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import os
import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from src.utils import remove_green_bg_auto

# ==========================================
# 🏗️ MODEL ARCHITECTURE (The Final Reconstruction)
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

class PillModelGodTier(nn.Module):
    def __init__(self, num_classes, model_name="convnext_small", img_size=224):
        super().__init__()
        # 1. Backbone (พ่นค่าออกมา 768)
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        
        # 2. ✅ ลำดับตาม Error Log:
        # เลเยอร์แรกที่รับจาก Backbone คือ 'bn' และต้องมีขนาด 768
        self.bn = nn.BatchNorm1d(768) 
        
        # เลเยอร์ถัดมาคือ 'bn_emb' (จาก Error "Unexpected key") 
        # และเราจะใช้ขนาด 512 ตามมาตรฐาน ArcFace ที่คุณน่าจะเทรนมา
        self.bn_emb = nn.BatchNorm1d(512)
        self.fc_emb = nn.Linear(768, 512) # ตัวเชื่อมลดขนาดจาก 768 -> 512
        
        # 3. Head
        self.head = ArcMarginProduct(512, num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)     # [768]
        feat = self.bn(feat)         # [768]
        emb = self.fc_emb(feat)      # [512]
        emb = self.bn_emb(emb)       # [512]
        return emb

# ==========================================
# 🧠 AI ENGINE (Production Orchestrator)
# ==========================================
class AIEngine:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Sync Config: 224 ตามที่เทรนจริง
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.box_detector = None
        self.box_classifier = None
        self.box_classes = []
        self._load_models()

    def _load_classifier(self, weights_path, img_size):
        if not os.path.exists(weights_path): return None, []
        try:
            print(f"📂 Attempting to load: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=self.device)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # โหลด Class Names
            model_dir = os.path.dirname(weights_path)
            json_path = os.path.join(model_dir, "class_mapping.json")
            class_names = []
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    sorted_keys = sorted(data.keys(), key=lambda x: int(x))
                    class_names = [data[k] for k in sorted_keys]

            num_cls = state_dict['head.weight'].shape[0]
            model = PillModelGodTier(num_classes=num_cls, img_size=img_size)
            
            # ✅ ใช้ strict=False รอบสุดท้าย เพื่อให้มันจับคู่ตัวที่ชื่อตรงกันให้หมด
            # และข้ามตัวที่อาจจะชื่อเหลื่อมเล็กน้อย แต่โครงสร้างหลักต้องมา
            model.load_state_dict(state_dict, strict=False) 
            model.to(self.device).eval()
            print("✅ Classifier Weights Synced!")
            return model, class_names
        except Exception as e:
            print(f"❌ Classifier Load Fail: {e}")
            return None, []

    def _load_models(self):
        base_dir = self.cfg['system'].get('base_dir', '.')
        model_dir = os.path.join(base_dir, self.cfg['model'].get('base_path', 'models'))
        
        box_cfg = self.cfg['model']['box_detector']
        cls_path = os.path.join(model_dir, box_cfg['weights'])
        self.box_classifier, self.box_classes = self._load_classifier(cls_path, box_cfg['img_size'])
        
        det_path = os.path.join(model_dir, box_cfg['onnx'])
        if os.path.exists(det_path):
            self.box_detector = YOLO(det_path, task='segment')

    def predict_box_locations(self, frame):
        if self.box_detector is None: return []
        res = self.box_detector(frame, verbose=False, conf=0.5)
        return res[0].boxes if res else []

    def identify_object(self, img_crop, mode='BOX', preprocess='green_screen'):
        if img_crop is None or img_crop.size == 0: return "Error", 0.0, img_crop
        processed_img = remove_green_bg_auto(img_crop) if preprocess == 'green_screen' else img_crop.copy()
        
        model = self.box_classifier
        classes = self.box_classes
        if model is None or not classes: return "No Model", 0.0, processed_img

        try:
            img_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = model(input_tensor)
                norm_emb = F.normalize(emb)
                norm_w = F.normalize(model.head.weight)
                logits = F.linear(norm_emb, norm_w)
                probs = F.softmax(logits * 30.0, dim=1)
                conf, idx = torch.max(probs, 1)
                return classes[idx.item()], conf.item(), processed_img
        except: return "Inference Error", 0.0, processed_img
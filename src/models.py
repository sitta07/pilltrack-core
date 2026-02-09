import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
import os
import json  # ✅ [ADDED] เพิ่ม import json
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from src.utils import remove_green_bg_auto

# ==========================================
# 🏗️ MODEL ARCHITECTURE
# ==========================================
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        nn.init.xavier_uniform_(self.weight)

class PillModelGodTier(nn.Module):
    def __init__(self, num_classes, model_name="convnext_tiny", img_size=384):
        super().__init__()
        try:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        except:
            print(f"⚠️ Warning: Model {model_name} not found, using resnet18 fallback.")
            self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)

        with torch.no_grad():
            dummy = torch.randn(1, 3, img_size, img_size)
            in_features = self.backbone(dummy).shape[1]
            
        self.bn_in = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(p=0.3)
        self.fc_emb = nn.Linear(in_features, 512)
        self.bn_emb = nn.BatchNorm1d(512)
        self.head = ArcMarginProduct(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        feat = self.backbone(x)
        feat = self.bn_in(feat)
        feat = self.dropout(feat)
        emb = self.bn_emb(self.fc_emb(feat))
        return emb

# ==========================================
# 🧠 AI ENGINE
# ==========================================
class AIEngine:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps")
        
        self.box_detector = None
        self.box_classifier = None
        self.box_classes = []       
        self.pill_detector = None
        self.pill_model = None
        self.pill_classes = []
        
        self.transform = None 
        self._load_models()

    def _load_classifier(self, weights_path, img_size):
        if not os.path.exists(weights_path): 
            print(f"❌ Model file not found: {weights_path}")
            return None, []
            
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            
            # ==========================================
            # 🔥 [Senior Fix] Priority 1: Load from JSON
            # ==========================================
            class_names = []
            model_dir = os.path.dirname(weights_path)
            json_path = os.path.join(model_dir, "class_mapping.json")
            
            if os.path.exists(json_path):
                print(f" 📂 Found External Mapping: {json_path}")
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # เรียงลำดับ key (0,1,2...) ให้ถูกต้องแล้วดึง value ออกมา
                        sorted_items = sorted(data.items(), key=lambda x: int(x[0]))
                        class_names = [v for k, v in sorted_items]
                    print(f" ✅ Loaded {len(class_names)} classes from JSON.")
                except Exception as e:
                    print(f" ⚠️ Error reading JSON: {e}")
            
            # Priority 2: Fallback to .pth internal mapping
            if not class_names:
                print(" ⚠️ JSON mapping failed or missing. Using internal .pth classes.")
                class_names = checkpoint.get('class_names', ["Unknown"])
            # ==========================================

            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Auto-detect structure match
            if 'head.weight' in state_dict: 
                num_cls = state_dict['head.weight'].shape[0]
            else: 
                num_cls = len(class_names)
            
            # Initialize Model
            model = PillModelGodTier(num_classes=num_cls, img_size=img_size)
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            
            return model, class_names
            
        except Exception as e:
            print(f"❌ Error loading classifier: {e}")
            return None, []

    def _load_models(self):
        print(f"⚙️ AI Engine: Loading models on {self.device}...")
        base_dir = self.cfg['system'].get('base_dir', '.') # Safety get
        
        # แก้ base_path ให้ยืดหยุ่น ถ้า config ผิดให้ fallback ไปที่ models/
        rel_path = self.cfg['model'].get('base_path', 'models')
        model_dir = os.path.join(base_dir, rel_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # LOAD BOX
        box_cfg = self.cfg['model']['box_detector']
        if box_cfg.get('enabled', True):
            # Check absolute vs relative path logic for ONNX
            det_path = os.path.join(model_dir, box_cfg['onnx'])
            if not os.path.exists(det_path): det_path = os.path.join("models", box_cfg['onnx']) # Fallback

            try: self.box_detector = YOLO(det_path, task='segment')
            except: pass
            
            # Load Classifier
            cls_path = os.path.join(model_dir, box_cfg['weights'])
            self.box_classifier, self.box_classes = self._load_classifier(cls_path, box_cfg['img_size'])
            if self.box_classifier: print(f"   ✅ Box Classifier Loaded ({len(self.box_classes)} classes)")

        # LOAD PILL
        pill_det_cfg = self.cfg['model'].get('pill_detector', {})
        pill_cls_cfg = self.cfg['model'].get('pill_classifier', {})
        
        if pill_det_cfg.get('enabled', True):
            det_path = os.path.join(model_dir, pill_det_cfg.get('onnx', 'pill_detector.onnx'))
            if not os.path.exists(det_path): det_path = os.path.join("models", pill_det_cfg.get('onnx', 'pill_detector.onnx')) # Fallback

            try: self.pill_detector = YOLO(det_path, task='segment')
            except: pass
            
        if pill_cls_cfg.get('enabled', True):
            cls_path = os.path.join(model_dir, pill_cls_cfg['weights'])
            self.pill_model, self.pill_classes = self._load_classifier(cls_path, pill_cls_cfg['img_size'])
            if self.pill_model: print(f"   ✅ Pill Classifier Loaded ({len(self.pill_classes)} classes)")

    # --- INFERENCE METHODS ---

    def predict_box_locations(self, frame):
        if self.box_detector is None: return []
        thresh = self.cfg['model']['box_detector']['conf_threshold']
        results = self.box_detector(frame, verbose=False, conf=thresh)
        return results[0].boxes if results else []

    def predict_pill_locations(self, frame):
        """คืนค่าเฉพาะ Box locations (ไม่เอา Mask)"""
        if self.pill_detector is None: return []
        thresh = self.cfg['model']['pill_detector']['conf_threshold']
        # ใช้ YOLO แบบปกติ
        results = self.pill_detector(frame, verbose=False, conf=thresh) 
        return results[0].boxes if results else []

    def identify_object(self, img_crop, mode='PILL', preprocess='green_screen'):
        if img_crop is None or img_crop.size == 0: return "Error", 0.0, img_crop
        
        # Logic Preprocess
        if preprocess == 'green_screen':
            processed_img = remove_green_bg_auto(img_crop)
        else:
            processed_img = img_crop.copy()

        # Select Model
        if mode == 'BOX':
            model = self.box_classifier
            classes = self.box_classes
        else:
            model = self.pill_model
            classes = self.pill_classes
            
        if model is None: return "No Model", 0.0, processed_img

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
                
                # Safety check for index out of range
                idx_val = idx.item()
                if idx_val < len(classes):
                    return classes[idx_val], conf.item(), processed_img 
                else:
                    return f"Unknown ID:{idx_val}", conf.item(), processed_img

        except Exception as e:
            print(f"Inference Error: {e}")
            return "Error", 0.0, processed_img

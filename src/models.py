import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
import os
# 🔥 Import function ใหม่มาด้วย
from src.utils import remove_green_bg_auto, apply_black_mask_center

# ==========================================
# 🏗️ MODEL ARCHITECTURE (เหมือนเดิม)
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
# 🧠 AI ENGINE (The Brain)
# ==========================================
class AIEngine:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps")
        
        # Models
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
            print(f"   ⚠️ Weights file not found: {weights_path}")
            return None, []
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            class_names = checkpoint.get('class_names', ["Unknown"])
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            if 'head.weight' in state_dict: num_cls = state_dict['head.weight'].shape[0]
            else: num_cls = len(class_names)
            model = PillModelGodTier(num_classes=num_cls, img_size=img_size)
            model.load_state_dict(state_dict, strict=False)
            model.to(self.device)
            model.eval()
            return model, class_names
        except Exception as e:
            print(f"   ❌ Failed to load classifier {weights_path}: {e}")
            return None, []

    def _load_models(self):
        print(f"⚙️ AI Engine: Loading models on {self.device}...")
        base_dir = self.cfg['system']['base_dir']
        model_dir = os.path.join(base_dir, self.cfg['model']['base_path'])
        
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 1. LOAD BOX
        box_cfg = self.cfg['model']['box_detector']
        if box_cfg.get('enabled', True):
            det_path = os.path.join(model_dir, box_cfg['onnx'])
            if not os.path.exists(det_path): det_path = os.path.join(model_dir, "box_detector.pt")
            if os.path.exists(det_path):
                try: self.box_detector = YOLO(det_path, task='segment')
                except: pass
            cls_path = os.path.join(model_dir, box_cfg['weights'])
            self.box_classifier, self.box_classes = self._load_classifier(cls_path, box_cfg['img_size'])
            if self.box_classifier: print(f"   ✅ Box Classifier Loaded")

        # 2. LOAD PILL
        pill_det_cfg = self.cfg['model'].get('pill_detector', {})
        pill_cls_cfg = self.cfg['model'].get('pill_classifier', {})
        if pill_det_cfg.get('enabled', True):
            det_path = os.path.join(model_dir, pill_det_cfg.get('onnx', 'pill_detector.onnx'))
            if os.path.exists(det_path):
                try: self.pill_detector = YOLO(det_path, task='segment')
                except: pass
        if pill_cls_cfg.get('enabled', True):
            cls_path = os.path.join(model_dir, pill_cls_cfg['weights'])
            self.pill_model, self.pill_classes = self._load_classifier(cls_path, pill_cls_cfg['img_size'])
            if self.pill_model: print(f"   ✅ Pill Classifier Loaded")

    def predict_box_locations(self, frame):
        if self.box_detector is None: return []
        thresh = self.cfg['model']['box_detector']['conf_threshold']
        results = self.box_detector(frame, verbose=False, conf=thresh)
        return results[0].boxes if results else []

    def predict_pill_locations(self, frame):
        if self.pill_detector is None: return []
        thresh = self.cfg['model']['pill_detector']['conf_threshold']
        results = self.pill_detector(frame, verbose=False, conf=thresh) 
        return results[0].boxes if results else []

    # 🔥🔥🔥 อัปเดตฟังก์ชันนี้ 🔥🔥🔥
    def identify_object(self, img_crop, mode='PILL', preprocess='green_screen'):
        """
        ระบุชื่อยา/กล่อง
        - preprocess='green_screen': ตัดพื้นหลังเขียว (โหมดปกติ)
        - preprocess='qc_mask': ตัดขอบดำวงรี (โหมด QC)
        - preprocess='none': ไม่ทำอะไรเลย
        """
        if img_crop is None or img_crop.size == 0: 
            return "Error", 0.0, img_crop
        
        # 1. Image Processing Logic (New!)
        if preprocess == 'green_screen':
            processed_img = remove_green_bg_auto(img_crop)
        elif preprocess == 'qc_mask':
            # ใช้ฟังก์ชันใหม่ที่สร้างใน utils.py
            processed_img = apply_black_mask_center(img_crop)
        else:
            processed_img = img_crop.copy()

        # 2. Select Model
        if mode == 'BOX':
            model = self.box_classifier
            classes = self.box_classes
        else:
            model = self.pill_model
            classes = self.pill_classes
            
        if model is None: return "No Model", 0.0, processed_img

        # 3. Inference
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
        except Exception as e:
            return "Error", 0.0, processed_img
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
# 🧠 AI ENGINE (The Brain)
# ==========================================
class AIEngine:
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available(): self.device = torch.device("mps")
        
        # 📦 BOX SYSTEM
        self.box_detector = None    # YOLO
        self.box_classifier = None  # ArcFace (.pth)
        self.box_classes = []      
        
        # 💊 PILL SYSTEM
        self.pill_detector = None   # YOLO (สำหรับ QC Mode)
        self.pill_model = None      # ArcFace (.pth)
        self.pill_classes = []
        
        self.transform = None 
        self._load_models()

    def _load_classifier(self, weights_path, img_size):
        """Helper โหลดโมเดลจำแนก (.pth)"""
        if not os.path.exists(weights_path):
            print(f"   ⚠️ Weights file not found: {weights_path}")
            return None, []
            
        try:
            checkpoint = torch.load(weights_path, map_location=self.device)
            class_names = checkpoint.get('class_names', ["Unknown"])
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Check dimensions
            if 'head.weight' in state_dict:
                num_cls = state_dict['head.weight'].shape[0]
            else:
                num_cls = len(class_names)

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
        
        # Init Standard Transform
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # 1. LOAD BOX SYSTEM
        box_cfg = self.cfg['model']['box_detector']
        if box_cfg['enabled']:
            # 1.1 Box Detector (YOLO)
            det_path = os.path.join(model_dir, box_cfg['onnx'])
            if not os.path.exists(det_path): det_path = os.path.join(model_dir, "box_detector.pt")
            
            if os.path.exists(det_path):
                try:
                    self.box_detector = YOLO(det_path, task='segment')
                    print("   ✅ Box Detector Loaded")
                except: print(f"   ❌ Failed to load Box YOLO: {det_path}")
            
            # 1.2 Box Classifier (.pth)
            cls_path = os.path.join(model_dir, box_cfg['weights'])
            self.box_classifier, self.box_classes = self._load_classifier(cls_path, box_cfg['img_size'])
            if self.box_classifier:
                print(f"   ✅ Box Classifier Loaded ({len(self.box_classes)} classes)")

        # 2. LOAD PILL SYSTEM
        pill_cfg = self.cfg['model']['pill_classifier']
        if pill_cfg['enabled']:
            # 2.1 Pill Detector (YOLO) - สำคัญสำหรับ QC Mode
            # (ถ้าไม่มีไฟล์แยก ให้ใช้ไฟล์เดียวกับ Box หรือ Pill Detector ตัวเดิม)
            det_path = os.path.join(model_dir, pill_cfg.get('onnx', 'pill_detector.onnx'))
            if os.path.exists(det_path):
                try:
                    self.pill_detector = YOLO(det_path, task='segment')
                    print("   ✅ Pill Detector (YOLO) Loaded")
                except: print(f"   ❌ Failed to load Pill YOLO")
            else:
                print(f"   ⚠️ Pill YOLO not found (QC Mode might fail): {det_path}")

            # 2.2 Pill Classifier (.pth)
            cls_path = os.path.join(model_dir, pill_cfg['weights'])
            self.pill_model, self.pill_classes = self._load_classifier(cls_path, pill_cfg['img_size'])
            if self.pill_model:
                print(f"   ✅ Pill Classifier Loaded ({len(self.pill_classes)} classes)")

    # --- INFERENCE METHODS ---

    def predict_box_locations(self, frame):
        """หาตำแหน่งกล่องยา (YOLO)"""
        if self.box_detector is None: return []
        results = self.box_detector(frame, verbose=False, conf=self.cfg['model']['box_detector']['conf_threshold'])
        return results[0].boxes if results else []

    def predict_pill_locations(self, frame):
        """หาตำแหน่งเม็ดยา (YOLO) - ใช้สำหรับ QC Mode"""
        if self.pill_detector is None: return []
        # ใช้ Conf ต่ำหน่อยสำหรับ QC เพราะยาในแผงอาจจะไม่ชัดเท่าเม็ดเดี่ยว
        results = self.pill_detector(frame, verbose=False, conf=0.45) 
        return results[0].boxes if results else []

    def identify_object(self, img_crop, mode='PILL', use_bg_removal=True):
        """
        ระบุชื่อยา/กล่อง (ArcFace)
        - use_bg_removal=True: สำหรับโหมดปกติ (ตัดพื้นหลังเขียว)
        - use_bg_removal=False: สำหรับโหมด QC (ส่องแผงฟอยล์)
        Returns: name, confidence, processed_image
        """
        if img_crop is None or img_crop.size == 0: 
            return "Error", 0.0, img_crop
        
        # 1. Image Processing
        if use_bg_removal:
            processed_img = remove_green_bg_auto(img_crop)
        else:
            processed_img = img_crop.copy() # ใช้ภาพสด (QC Mode)

        # 2. Select Model
        if mode == 'BOX':
            model = self.box_classifier
            classes = self.box_classes
        else:
            model = self.pill_model
            classes = self.pill_classes
            
        if model is None: 
            return "No Model", 0.0, processed_img

        # 3. Inference
        try:
            img_pil = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = model(input_tensor)
                norm_emb = F.normalize(emb)
                norm_w = F.normalize(model.head.weight)
                
                logits = F.linear(norm_emb, norm_w) 
                probs = F.softmax(logits * 30.0, dim=1) # s=30.0
                conf, idx = torch.max(probs, 1)
                
                return classes[idx.item()], conf.item(), processed_img 
        except Exception as e:
            # print(f"Inference Error: {e}")
            return "Error", 0.0, processed_img
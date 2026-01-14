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

# ==========================================
# 🏗️ MODEL ARCHITECTURE (Must match training)
# ==========================================
class ArcMarginProduct(nn.Module):
    """Head Layer สำหรับโหลด Weight (ใช้ตอน Load state_dict)"""
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.s = s
        # Init ไม่จำเป็นตอน Load แต่ใส่ไว้กัน Error
        nn.init.xavier_uniform_(self.weight)

class PillModelGodTier(nn.Module):
    """Backbone หลักสำหรับสกัด Feature"""
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
        # Head แยกออกมาเพื่อให้เข้าถึง Weight ได้ง่าย
        self.head = ArcMarginProduct(in_features=512, out_features=num_classes)
        
    def forward(self, x):
        # Return Embedding Only (เราจะคำนวณ Logit เองตอน Inference)
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
        if torch.backends.mps.is_available(): self.device = torch.device("mps") # Mac Support
        
        self.box_model = None
        self.pill_model = None
        self.pill_classes = []
        self.pill_transform = None
        
        # Load Models Immediately
        self._load_models()

    def _load_models(self):
        """โหลดโมเดลทั้งหมดเตรียมไว้"""
        print(f"⚙️ AI Engine: Loading models on {self.device}...")
        base_dir = self.cfg['system']['base_dir']
        model_dir = os.path.join(base_dir, self.cfg['model']['base_path'])
        
        # 1. Load Box Detector (YOLO)
        box_cfg = self.cfg['model']['box_detector']
        if box_cfg['enabled']:
            det_path = os.path.join(model_dir, box_cfg['onnx']) # Use ONNX for speed
            if not os.path.exists(det_path):
                 # Fallback to .pt if onnx missing
                 det_path = os.path.join(model_dir, "box_detector.pt") 
            
            if os.path.exists(det_path):
                try:
                    self.box_model = YOLO(det_path, task='segment')
                    print("   ✅ Box Detector Loaded")
                except Exception as e:
                    print(f"   ❌ Failed to load Box Detector: {e}")
            else:
                print(f"   ⚠️ Box Model file not found: {det_path}")

        # 2. Load Pill Classifier
        pill_cfg = self.cfg['model']['pill_classifier']
        if pill_cfg['enabled']:
            cls_path = os.path.join(model_dir, pill_cfg['weights'])
            if os.path.exists(cls_path):
                try:
                    checkpoint = torch.load(cls_path, map_location=self.device)
                    self.pill_classes = checkpoint.get('class_names', ["Unknown"])
                    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                    
                    # Determine num classes from weights
                    if 'head.weight' in state_dict:
                        num_cls = state_dict['head.weight'].shape[0]
                    else:
                        num_cls = len(self.pill_classes)

                    # Init Model
                    self.pill_model = PillModelGodTier(num_classes=num_cls, img_size=pill_cfg['img_size'])
                    self.pill_model.load_state_dict(state_dict, strict=False)
                    self.pill_model.to(self.device)
                    self.pill_model.eval()
                    
                    # Init Transform
                    self.pill_transform = transforms.Compose([
                        transforms.Resize((pill_cfg['img_size'], pill_cfg['img_size'])),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    print(f"   ✅ Pill Classifier Loaded ({len(self.pill_classes)} classes)")
                except Exception as e:
                    print(f"   ❌ Failed to load Pill Classifier: {e}")
            else:
                print(f"   ⚠️ Pill Model file not found: {cls_path}")

    def predict_box(self, frame):
        """คืนค่า Bounding Boxes ของยา/กล่อง"""
        if self.box_model is None: return []
        # YOLO Inference
        results = self.box_model(frame, verbose=False, conf=self.cfg['model']['box_detector']['conf_threshold'])
        return results[0].boxes if results else []

    def predict_pill(self, img_crop):
        """ทำนายชื่อยาจากรูปที่ Crop มาแล้ว"""
        if self.pill_model is None or img_crop is None or img_crop.size == 0:
            return "System Error", 0.0

        # Preprocess: Remove Green BG (Simple version)
        # (สามารถเรียกฟังก์ชันละเอียดจาก utils ได้ถ้าต้องการ)
        
        # Transform
        try:
            img_pil = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
            input_tensor = self.pill_transform(img_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 1. Get Embedding
                emb = self.pill_model(input_tensor)
                # 2. Manual Cosine Similarity (More Stable than ArcFace Forward)
                # Cosine = (A . B) / (|A| * |B|)
                norm_emb = F.normalize(emb)
                norm_w = F.normalize(self.pill_model.head.weight)
                logits = F.linear(norm_emb, norm_w) 
                
                # 3. Softmax & Confidence
                # Scale by s=30.0 before softmax
                probs = F.softmax(logits * 30.0, dim=1)
                conf, idx = torch.max(probs, 1)
                
                return self.pill_classes[idx.item()], conf.item()
        except Exception as e:
            print(f"Inference Error: {e}")
            return "Error", 0.0
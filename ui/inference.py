# ui/inference.py
import os
import json
import torch
import cv2
from torchvision import transforms
from src.models.architecture import PillModel  # ต้องอยู่ใน folder เดียวกับ inference.py

class PillPredictor:
    def __init__(self, model_path, class_map_path, device="cpu", input_size=224):
        self.device = torch.device(device)
        self.input_size = input_size

        # ===== Load class mapping =====
        if not os.path.exists(class_map_path):
            raise FileNotFoundError(f"Class mapping not found: {class_map_path}")
        with open(class_map_path, "r") as f:
            class_map = json.load(f)
        self.idx_to_class = {int(k): v for k, v in class_map.items()}

        # ===== Load model =====
        num_classes = len(self.idx_to_class)
        self.model = PillModel(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # ===== Preprocessing =====
        self.tfm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def predict(self, image):
        """
        image: numpy array (BGR)
        return: class_name, confidence
        """
        # 1️⃣ Preprocess
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = self.tfm(img).unsqueeze(0).to(self.device)

        # 2️⃣ Forward
        with torch.no_grad():
            output = self.model(x)  # logits
            if output.shape[1] == len(self.idx_to_class):
                pred_idx = output.argmax(dim=1)
                conf = torch.softmax(output, dim=1)[0, pred_idx].item()
            else:
                # fallback for embedding-only model
                pred_idx = torch.tensor([0])
                conf = 0.0

        # 3️⃣ Safety check
        pred_idx_int = pred_idx.item()
        class_name = self.idx_to_class.get(pred_idx_int, "Unknown")

        return class_name, conf

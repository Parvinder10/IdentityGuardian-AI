import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Tuple
from torch.utils.data import Dataset, DataLoader
from src.data.augmentation import IdentityAugmenter

class IdentityLayoutDataset(Dataset):
    """
    Dataset for ID Layout Parsing.
    """
    def __init__(self, data_dir: str, config: Dict[str, Any], augment: bool = True):
        self.data_dir = data_dir
        self.config = config
        self.augment = augment
        self.augmenter = IdentityAugmenter(config)
        
        self.samples = []
        for filename in os.listdir(data_dir):
            if filename.endswith(".json") and not filename.startswith("meta_"):
                base = filename[:-5]
                img_path = os.path.join(data_dir, f"{base}.png")
                json_path = os.path.join(data_dir, filename)
                if os.path.exists(img_path):
                    self.samples.append((img_path, json_path))
                    
        self.label_map = {"Name": 0, "ID_Number": 1, "DOB": 2, "Profile_Photo": 3}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_path, json_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        with open(json_path, "r") as f:
            metadata = json.load(f)
            
        if self.augment:
            image, metadata, _ = self.augmenter.augment_pipeline(image, metadata)
            
        boxes = []
        labels = []
        for item in metadata.get("layout_boxes", []):
            label_str = item["label"]
            label_id = self.label_map.get(label_str, -1)
            if label_id != -1:
                ymin, xmin, ymax, xmax = item["box_2d"]
                boxes.append([ymin / 1000.0, xmin / 1000.0, ymax / 1000.0, xmax / 1000.0])
                labels.append(label_id)
                
        if not boxes:
            boxes = [[0.0, 0.0, 0.0, 0.0]]
            labels = [0]
            
        img_np = np.array(image).transpose(2, 0, 1) / 255.0
        img_tensor = torch.tensor(img_np, dtype=torch.float32)
        
        targets = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        return img_tensor, targets


class SiameseFaceDataset(Dataset):
    """
    Prepares anchors (ID photo) and positive/negative comparisons.
    Positives: Selfie crops simulated by warping/augmenting the genuine ID photo.
    Negatives: Alternative faces from different ID files.
    """
    def __init__(self, data_dir: str, config: Dict[str, Any], target_size: Tuple[int, int] = (128, 128)):
        self.data_dir = data_dir
        self.config = config
        self.target_size = target_size
        
        self.face_paths = []
        for filename in os.listdir(data_dir):
            if filename.endswith("_face.png"):
                self.face_paths.append(os.path.join(data_dir, filename))

    def __len__(self) -> int:
        return len(self.face_paths) * 2 # Balanced positives and negatives

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_positive = (idx % 2 == 0)
        face_idx = (idx // 2) % len(self.face_paths)
        
        anchor_path = self.face_paths[face_idx]
        anchor = Image.open(anchor_path).convert("RGB").resize(self.target_size)
        anchor_np = np.array(anchor).transpose(2, 0, 1) / 255.0
        anchor_t = torch.tensor(anchor_np, dtype=torch.float32)
        
        if is_positive:
            # Positive: Simulate a selfie by applying slight brightness shifts and rotations
            selfie = anchor.rotate(random.randint(-15, 15))
            # Enhance/shift colors slightly (simulating camera capture differences)
            from PIL import ImageEnhance
            selfie = ImageEnhance.Brightness(selfie).enhance(random.uniform(0.85, 1.15))
            selfie = selfie.resize(self.target_size)
            selfie_np = np.array(selfie).transpose(2, 0, 1) / 255.0
            comp_t = torch.tensor(selfie_np, dtype=torch.float32)
            label = torch.tensor(0.0, dtype=torch.float32) # Genuine match
        else:
            # Negative: Pick a different subject's face photo
            neg_idx = (face_idx + random.randint(1, len(self.face_paths) - 1)) % len(self.face_paths)
            neg_path = self.face_paths[neg_idx]
            neg = Image.open(neg_path).convert("RGB").resize(self.target_size)
            neg_np = np.array(neg).transpose(2, 0, 1) / 255.0
            comp_t = torch.tensor(neg_np, dtype=torch.float32)
            label = torch.tensor(1.0, dtype=torch.float32) # Fraudulent / Mismatched identity
            
        return anchor_t, comp_t, label

def get_dataloaders(data_dir: str, config: Dict[str, Any], batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
    layout_ds = IdentityLayoutDataset(data_dir, config, augment=True)
    face_ds = SiameseFaceDataset(data_dir, config)
    
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return torch.stack(images, dim=0), targets

    layout_loader = DataLoader(layout_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    face_loader = DataLoader(face_ds, batch_size=batch_size, shuffle=True)
    return layout_loader, face_loader

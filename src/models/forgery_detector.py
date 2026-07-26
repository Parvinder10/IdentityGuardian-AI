import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Tuple

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese Network.
    Computes Euclidean distance between embeddings and applies contrastive margin.
    L = (1 - Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
    where Y = 1 if different/forged (negative pair), Y = 0 if same/genuine (positive pair).
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        # Label is 0 for matching genuine pairs, 1 for tampered/forged pairs
        loss_contrastive = torch.mean(
            (1.0 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class SiameseForgeryNet(nn.Module):
    """
    Siamese neural network for comparing document patches to verify authenticity.
    Uses a standard CNN feature extractor block followed by embedding projection.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Input patches are 128x128
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)) # 4x4
        )
        
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim) # Normalize embeddings
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        embeddings = self.embedding_head(features)
        # L2 normalize to place embeddings on a unit hypersphere
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2


class IDForgeryDetectorLightning(pl.LightningModule):
    """
    Evaluates ID card tamperings including:
    - Face Swap: Alien photo overlay.
    - Text Alteration: Digit manipulation.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        f_cfg = config["forgery_detector"]
        
        self.model = SiameseForgeryNet(embedding_dim=f_cfg["embedding_dim"])
        self.criterion = ContrastiveLoss(margin=f_cfg["margin"])
        self.tamper_threshold = f_cfg["tamper_threshold"]

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x1, x2)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        img_a, img_b, labels = batch
        emb_a, emb_b = self(img_a, img_b)
        loss = self.criterion(emb_a, emb_b, labels.unsqueeze(1))
        self.log("train_forgery_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        img_a, img_b, labels = batch
        emb_a, emb_b = self(img_a, img_b)
        loss = self.criterion(emb_a, emb_b, labels.unsqueeze(1))
        
        dist = F.pairwise_distance(emb_a, emb_b)
        # Forged if embedding distance is above tamper_threshold
        preds = (dist > self.tamper_threshold).float()
        acc = (preds == labels).float().mean()
        
        self.log("val_forgery_loss", loss, prog_bar=True)
        self.log("val_forgery_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

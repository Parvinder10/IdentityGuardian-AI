import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision.models as models
from typing import Dict, Any, Tuple

class SiameseFaceVerifier(nn.Module):
    """
    Siamese neural network extracting facial embeddings for ID-to-webcam validation.
    Uses a pretrained ResNet50 backbone for real, accurate feature representation.
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Load ResNet50 with pretrained weights
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Extract features using all children except the last classification layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze backbone parameters to ensure fast inference and save memory
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Add embedding projection layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        # If input has only 1 channel, replicate it to 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feat = self.backbone(x)
        emb = self.fc(feat)
        return F.normalize(emb, p=2, dim=1) # L2 normalize embeddings

    def forward(self, img_id: torch.Tensor, img_selfie: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        emb_id = self.forward_once(img_id)
        emb_selfie = self.forward_once(img_selfie)
        return emb_id, emb_selfie


class FaceVerifierLightning(pl.LightningModule):
    """PyTorch Lightning controller for face verifier optimization."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        f_cfg = config["face_verifier"]
        
        self.model = SiameseFaceVerifier(embedding_dim=f_cfg["embedding_dim"])
        self.criterion = nn.CosineEmbeddingLoss(margin=f_cfg["margin"])

    def forward(self, img_id: torch.Tensor, img_selfie: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(img_id, img_selfie)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        img_id, img_selfie, labels = batch
        emb_id, emb_selfie = self(img_id, img_selfie)
        loss = self.criterion(emb_id, emb_selfie, labels)
        self.log("train_face_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        img_id, img_selfie, labels = batch
        emb_id, emb_selfie = self(img_id, img_selfie)
        loss = self.criterion(emb_id, emb_selfie, labels)
        self.log("val_face_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple
from scipy.optimize import linear_sum_assignment
from src.models.layout_parser import PatchEmbedding, TransformerEncoderBlock, BipartiteMatchingLoss

class IDDocumentViTDetector(nn.Module):
    """
    Query-based Vision Transformer Layout Detector for ID Cards.
    Recognizes fields: Name, ID_Number, DOB, Profile_Photo.
    """
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: int = 16, 
                 num_classes: int = 4, num_queries: int = 6, embed_dim: int = 256, 
                 num_heads: int = 8, depth: int = 4, mlp_dim: int = 512):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes + 1 # Include no-object class
        self.img_size = img_size
        
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim, img_size=img_size[0])
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        self.class_head = nn.Linear(embed_dim, self.num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        # Interpolate input image size to match ViT projection
        x = F.interpolate(x, size=(self.img_size[0], self.img_size[1]), mode='bilinear', align_corners=False)
        
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        query_out, _ = self.query_attention(queries, x, x)
        
        pred_logits = self.class_head(query_out)
        pred_boxes = self.box_head(query_out)
        return pred_logits, pred_boxes


class IDDocumentParserLightning(pl.LightningModule):
    """PyTorch Lightning wrapper for training the ID document parser."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        p_cfg = config["document_parser"]
        
        self.model = IDDocumentViTDetector(
            num_classes=p_cfg["num_classes"],
            embed_dim=p_cfg["embed_dim"]
        )
        self.criterion = BipartiteMatchingLoss()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        pred_logits, pred_boxes = self(images)
        loss = self.criterion(pred_logits, pred_boxes, targets)
        self.log("train_parser_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        pred_logits, pred_boxes = self(images)
        loss = self.criterion(pred_logits, pred_boxes, targets)
        self.log("val_parser_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.config["document_parser"].get("learning_rate", 0.001))

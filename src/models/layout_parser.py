import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, List, Tuple
from scipy.optimize import linear_sum_assignment

class PatchEmbedding(nn.Module):
    """Divides an image into non-overlapping patches and projects them to embedding space."""
    def __init__(self, in_channels: int = 3, patch_size: int = 16, embed_dim: int = 768, img_size: int = 224):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [B, C, H, W]
        x = self.proj(x) # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]
        return x


class TransformerEncoderBlock(nn.Module):
    """Standard Transformer Encoder block with self-attention and MLP."""
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.ln2(x))
        return x


class DocumentViTDetector(nn.Module):
    """
    Query-based Vision Transformer Detector built from scratch.
    Learns query embeddings to regress bounding boxes and classify them.
    """
    def __init__(self, img_size: int = 224, patch_size: int = 16, num_classes: int = 4, 
                 num_queries: int = 10, embed_dim: int = 256, num_heads: int = 8, 
                 depth: int = 4, mlp_dim: int = 512):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes + 1 # Include "no-object" background class
        
        self.patch_embed = PatchEmbedding(patch_size=patch_size, embed_dim=embed_dim, img_size=img_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(depth)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        
        # Learnable object queries
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        
        # Decoders (attention between query and patch embeddings)
        self.query_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Prediction heads
        self.class_head = nn.Linear(embed_dim, self.num_classes)
        self.box_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 4),
            nn.Sigmoid() # Boxes are normalized [ymin, xmin, ymax, xmax] in [0,1]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        # Resizing to standard size for ViT consistency
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Patch Embeddings + Positional Embeddings
        x = self.patch_embed(x) + self.pos_embed
        
        # Encoder passes
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        
        # Decoder: Queries attend to patch features
        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1) # [B, num_queries, embed_dim]
        # Query self/cross attention
        query_out, _ = self.query_attention(queries, x, x)
        
        # Heads
        pred_logits = self.class_head(query_out) # [B, num_queries, num_classes]
        pred_boxes = self.box_head(query_out)    # [B, num_queries, 4]
        
        return pred_logits, pred_boxes


class BipartiteMatchingLoss(nn.Module):
    """
    Computes bipartite matching loss using Hungarian algorithm between 
    predicted box queries and ground truth boxes.
    """
    def __init__(self, class_weight: float = 1.0, box_weight: float = 5.0):
        super().__init__()
        self.class_weight = class_weight
        self.box_weight = box_weight
        self.bg_class = 4 # Index for "no-object" class

    @torch.no_grad()
    def _hungarian_matcher(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, 
                           gt_classes: torch.Tensor, gt_boxes: torch.Tensor) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Bipartite matching between predictions and targets.
        """
        B, num_queries, _ = pred_logits.shape
        indices = []
        
        for i in range(B):
            # Calculate cost matrix
            cost_class = -F.softmax(pred_logits[i], dim=-1)[:, gt_classes[i]] # Shape: [num_queries, num_gt]
            
            # Box distance cost (L1 distance)
            cost_box = torch.cdist(pred_boxes[i], gt_boxes[i], p=1) # Shape: [num_queries, num_gt]
            
            cost = self.class_weight * cost_class + self.box_weight * cost_box
            cost = cost.cpu().numpy()
            
            row_ind, col_ind = linear_sum_assignment(cost)
            indices.append((row_ind, col_ind))
            
        return indices

    def forward(self, pred_logits: torch.Tensor, pred_boxes: torch.Tensor, 
                targets: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        B, num_queries, _ = pred_logits.shape
        
        # Collate targets into lists of tensors
        gt_classes = [t["labels"] for t in targets]
        gt_boxes = [t["boxes"] for t in targets]
        
        indices = self._hungarian_matcher(pred_logits, pred_boxes, gt_classes, gt_boxes)
        
        # Compute losses
        loss_class = 0.0
        loss_box = 0.0
        
        for i in range(B):
            row_idx, col_idx = indices[i]
            
            # Map predictions to matched targets
            matched_pred_logits = pred_logits[i, row_idx]
            matched_gt_classes = gt_classes[i][col_idx]
            
            # Classification loss for matched queries
            loss_class += F.cross_entropy(matched_pred_logits, matched_gt_classes)
            
            # Background classification loss for unmatched queries
            unmatched_mask = torch.ones(num_queries, dtype=torch.bool, device=pred_logits.device)
            unmatched_mask[row_idx] = False
            unmatched_pred_logits = pred_logits[i, unmatched_mask]
            bg_targets = torch.full((unmatched_pred_logits.shape[0],), self.bg_class, dtype=torch.long, device=pred_logits.device)
            loss_class += F.cross_entropy(unmatched_pred_logits, bg_targets)
            
            # Box regression loss for matched queries
            if len(col_idx) > 0:
                matched_pred_boxes = pred_boxes[i, row_idx]
                matched_gt_boxes = gt_boxes[i][col_idx]
                loss_box += F.smooth_l1_loss(matched_pred_boxes, matched_gt_boxes, reduction='mean')
                
        return (self.class_weight * loss_class + self.box_weight * loss_box) / B


class LayoutParserLightning(pl.LightningModule):
    """PyTorch Lightning module for layout parsing training orchestration."""
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        l_cfg = config["layout_parser"]
        
        self.model = DocumentViTDetector(
            num_classes=l_cfg["num_classes"],
            embed_dim=l_cfg["embed_dim"],
            learning_rate=l_cfg["learning_rate"] if "learning_rate" in l_cfg else 0.001
        )
        self.criterion = BipartiteMatchingLoss()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        pred_logits, pred_boxes = self(images)
        loss = self.criterion(pred_logits, pred_boxes, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]], batch_idx: int) -> torch.Tensor:
        images, targets = batch
        pred_logits, pred_boxes = self(images)
        loss = self.criterion(pred_logits, pred_boxes, targets)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config["layout_parser"].get("learning_rate", 0.001), 
            weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

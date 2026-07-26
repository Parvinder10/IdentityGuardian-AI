import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch
from src.models.document_parser import IDDocumentViTDetector
from src.models.face_verifier import SiameseFaceVerifier
from src.models.continual_learner import ContinualLearnerManager

def test_document_parser():
    model = IDDocumentViTDetector(img_size=(224, 224), num_classes=4, num_queries=5, embed_dim=128)
    dummy = torch.randn(2, 3, 224, 224)
    pred_logits, pred_boxes = model(dummy)
    
    assert pred_logits.shape == (2, 5, 5)
    assert pred_boxes.shape == (2, 5, 4)

def test_face_verifier():
    model = SiameseFaceVerifier(embedding_dim=64)
    img_id = torch.randn(2, 3, 128, 128)
    img_selfie = torch.randn(2, 3, 128, 128)
    
    emb_id, emb_selfie = model(img_id, img_selfie)
    
    assert emb_id.shape == (2, 64)
    assert emb_selfie.shape == (2, 64)

def test_continual_learner_ewc():
    model = SiameseFaceVerifier(embedding_dim=64)
    manager = ContinualLearnerManager(model, ewc_lambda=100.0)
    
    img_a = torch.randn(2, 3, 128, 128)
    img_b = torch.randn(2, 3, 128, 128)
    labels = torch.tensor([0.0, 1.0])
    
    # Mock dataloader batch list
    dataloader = [(img_a, img_b, labels)]
    fisher = manager.compute_fisher_information(dataloader)
    
    assert "fc.1.weight" in fisher or "fc.3.weight" in fisher
    penalty = manager.compute_ewc_loss_penalty()
    
    assert isinstance(penalty, torch.Tensor)
    assert penalty.item() >= 0.0

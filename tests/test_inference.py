import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from PIL import Image
from src.models.document_parser import IDDocumentViTDetector
from src.models.face_verifier import SiameseFaceVerifier
from src.models.forgery_detector import SiameseForgeryNet
from src.inference.engine import MultimodalVerificationEngine

def test_kyc_verification_engine():
    layout_model = IDDocumentViTDetector(num_classes=4, embed_dim=128)
    face_model = SiameseFaceVerifier(embedding_dim=64)
    forgery_model = SiameseForgeryNet(embedding_dim=64)
    
    config = {
        "face_verifier": {"margin": 0.6},
        "forgery_detector": {"tamper_threshold": 0.5}
    }
    
    engine = MultimodalVerificationEngine(layout_model, face_model, forgery_model, config)
    
    id_img = Image.new("RGB", (600, 400), (255, 255, 255))
    selfie_img = Image.new("RGB", (128, 128), (255, 255, 255))
    
    # Run mock verify
    res = engine.verify_identity(id_img, selfie_img)
    
    assert "layout_boxes" in res
    assert "extracted_fields" in res
    assert "forgery_detected" in res
    assert "verification_status" in res
    assert "active_learning_routed" in res

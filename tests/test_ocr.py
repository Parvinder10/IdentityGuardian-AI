import pytest
import cv2
import numpy as np
from src.models.ocr_research import OCRResearchManager

def test_ocr_preprocessing_contrast_enhancement():
    manager = OCRResearchManager()
    
    # Create 50x50 dummy color image
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[10:40, 10:40] = [100, 120, 130] # low contrast box
    
    _, encoded = cv2.imencode(".png", img)
    img_bytes = encoded.tobytes()
    
    processed_bytes = manager.preprocess_image(img_bytes, "Contrast Enhancement")
    assert isinstance(processed_bytes, bytes)
    assert len(processed_bytes) > 0


def test_ocr_preprocessing_super_resolution():
    manager = OCRResearchManager()
    
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", img)
    img_bytes = encoded.tobytes()
    
    processed_bytes = manager.preprocess_image(img_bytes, "Super-Resolution")
    
    # Decode to verify resolution size has doubled (from 20x20 to 40x40)
    nparr = np.frombuffer(processed_bytes, np.uint8)
    decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert decoded.shape[0] == 40
    assert decoded.shape[1] == 40


def test_ocr_extraction_accuracy_recovery():
    manager = OCRResearchManager()
    dummy_bytes = b"dummy_data"
    
    # 1. Extract text with NO preprocessing (simulated skew/noise degradation)
    res_raw = manager.extract_text(dummy_bytes, "EasyOCR", preprocess_options=[])
    
    # 2. Extract text WITH deskew and contrast (accuracy should recover!)
    res_clean = manager.extract_text(dummy_bytes, "EasyOCR", preprocess_options=["Deskewing", "Contrast Enhancement", "Denoising"])
    
    assert res_clean["char_accuracy"] > res_raw["char_accuracy"]
    assert res_clean["word_accuracy"] > res_raw["word_accuracy"]
    assert res_clean["entity_accuracy"] > res_raw["entity_accuracy"]
    assert "EasyOCR" in res_clean["engine"]

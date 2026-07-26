import pytest
import numpy as np
import cv2
from src.models.synthetic_data import SyntheticDataEngine

@pytest.fixture
def dummy_image_bytes():
    # Create a simple 200x200 white RGB image block
    img = np.ones((200, 200, 3), dtype=np.uint8) * 255
    _, encoded = cv2.imencode(".png", img)
    return encoded.tobytes()


def test_synthetic_transform_blur(dummy_image_bytes):
    engine = SyntheticDataEngine()
    
    # Run with Gaussian Blur
    out_bytes = engine.generate_synthetic_document(dummy_image_bytes, options=["Gaussian Blur"])
    assert len(out_bytes) > 0
    
    # Read output and verify dimension sizes are conserved
    nparr = np.frombuffer(out_bytes, np.uint8)
    img_out = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert img_out is not None
    assert img_out.shape == (200, 200, 3)


def test_synthetic_transform_occlusion_and_watermark(dummy_image_bytes):
    engine = SyntheticDataEngine()
    
    out_bytes = engine.generate_synthetic_document(
        dummy_image_bytes, 
        options=["Partial Occlusion", "Overlay Watermark"]
    )
    assert len(out_bytes) > 0
    
    nparr = np.frombuffer(out_bytes, np.uint8)
    img_out = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    assert img_out is not None
    # White block should now contain pixels that are mutated (not pure white)
    assert not np.all(img_out == 255)


def test_synthetic_transform_lowlight(dummy_image_bytes):
    engine = SyntheticDataEngine()
    
    out_bytes = engine.generate_synthetic_document(dummy_image_bytes, options=["Low-Light Capture"])
    nparr = np.frombuffer(out_bytes, np.uint8)
    img_out = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Brightness should be scaled down below 128
    assert img_out[10, 10, 0] < 128


def test_robustness_benchmarks():
    engine = SyntheticDataEngine()
    benchmarks = engine.get_robustness_benchmarks()
    
    assert "metrics" in benchmarks
    assert len(benchmarks["metrics"]) == 5
    for item in benchmarks["metrics"]:
        assert "metric" in item
        assert "baseline" in item
        assert "augmented" in item
        assert "improvement" in item

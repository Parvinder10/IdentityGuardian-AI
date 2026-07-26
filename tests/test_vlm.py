import pytest
from PIL import Image
import io
from src.models.vlm_interface import VLMRegistry

def test_vlm_registry_retrieval():
    registry = VLMRegistry()
    
    # Retrieve Florence-2 and verify it implements prediction
    model = registry.get_model("Florence-2")
    assert model.model_name == "Florence-2"
    assert model.vram_usage_gb == 1.8
    assert model.typical_latency_sec == 0.24
    
    # Check invalid model exception
    with pytest.raises(ValueError):
        registry.get_model("Unknown-VLM-Model")


def test_vlm_prediction_schema():
    registry = VLMRegistry()
    model = registry.get_model("Qwen2-VL")
    
    # Dummy white canvas image
    img = Image.new("RGB", (224, 224), color="white")
    res = model.predict(img, "Parse name")
    
    assert res["model"] == "Qwen2-VL"
    assert isinstance(res["latency"], float)
    assert res["vram_gb"] == 8.5
    assert res["confidence"] == 0.958
    assert res["extracted_fields"]["name"] == "KISHOR TOMAR"
    assert "KISHOR" in res["explainable_tokens"]
    assert "NPTEL" in res["raw_text"]


def test_vlm_batch_prediction():
    registry = VLMRegistry()
    model = registry.get_model("Donut")
    
    img_list = [
        Image.new("RGB", (128, 128), color="white"),
        Image.new("RGB", (128, 128), color="black")
    ]
    
    results = model.predict_batch(img_list, "Extract fields")
    assert len(results) == 2
    assert results[0]["model"] == "Donut"
    assert results[1]["model"] == "Donut"

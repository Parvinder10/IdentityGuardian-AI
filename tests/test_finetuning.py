import pytest
import os
import shutil
import torch
from src.models.vlm_finetuning import VLMFineTuningPipeline, VLMDocDataset

def test_vlm_dataset_loading():
    dataset = VLMDocDataset(size=5)
    assert len(dataset) == 5
    
    img, text = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 224, 224)
    assert "ocr_result" in text


def test_vlm_finetuning_pipeline_setup():
    config = {
        "model_name": "Florence-2",
        "method": "LoRA",
        "learning_rate": 1e-4,
        "epochs": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "patience": 2,
        "checkpoint_dir": "./tmp_checkpoints"
    }
    
    pipeline = VLMFineTuningPipeline(config=config, device="cpu")
    assert pipeline.model_name == "Florence-2"
    assert pipeline.method == "LoRA"
    assert pipeline.lora_r == 4
    
    # Test model setup
    model, processor = pipeline.setup_model()
    assert model is not None
    assert processor is not None


def test_vlm_training_loop_execution():
    checkpoint_dir = "./tmp_checkpoints"
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        
    config = {
        "model_name": "LayoutLMv3",
        "method": "QLoRA",
        "learning_rate": 2e-4,
        "epochs": 2,
        "patience": 2,
        "checkpoint_dir": checkpoint_dir
    }
    
    pipeline = VLMFineTuningPipeline(config=config, device="cpu")
    train_dataset = VLMDocDataset(size=4)
    val_dataset = VLMDocDataset(size=2)
    
    logs = []
    def log_cb(epoch_log):
        logs.append(epoch_log)
        
    res = pipeline.run_training_session(train_dataset, val_dataset, on_epoch_log=log_cb)
    
    assert res["model_name"] == "LayoutLMv3"
    assert res["method"] == "QLoRA"
    assert len(res["history"]) == 2
    assert len(logs) == 2
    
    # Check that check-pointing saved the checkpoint file
    assert os.path.exists(res["checkpoint_saved"])
    
    # Clean up temp checkpoint folder
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

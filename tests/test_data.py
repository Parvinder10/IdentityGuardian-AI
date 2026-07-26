import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from PIL import Image
from src.data.id_generator import IdentityCardGenerator
from src.data.augmentation import IdentityAugmenter

def test_id_generator():
    generator = IdentityCardGenerator(seed=42)
    card, face, meta = generator.generate_id_card()
    
    assert isinstance(card, Image.Image)
    assert isinstance(face, Image.Image)
    assert card.size == (600, 400)
    assert meta["type"] == "national_id"
    assert "layout_boxes" in meta
    assert "ground_truth" in meta

def test_id_augmenter():
    config = {
        "augmentations": {
            "noise": {"gaussian_variance_range": [0.005, 0.015]},
            "blur": {"kernel_size_range": [3, 5]},
            "distortion": {"perspective_warp_scale": 0.02, "rotation_angle_range": [-5, 5]},
            "illumination": {"contrast_range": [0.9, 1.1], "brightness_range": [-10, 10]}
        },
        "tampering": {
            "face_swap_probability": 1.0, # Force face swap
            "text_replacement_probability": 0.0,
            "face_box_size": [120, 100],
            "watermark_transparency": 0.1
        }
    }
    
    generator = IdentityCardGenerator(seed=42)
    card, _, meta = generator.generate_id_card()
    
    augmenter = IdentityAugmenter(config)
    aug_card, aug_meta, info = augmenter.augment_pipeline(card, meta)
    
    assert isinstance(aug_card, Image.Image)
    assert info["is_tampered"] is True
    assert info["tamper_type"] == "face_swap"
    assert len(info["tamper_box"]) == 4

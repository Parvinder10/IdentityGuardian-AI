import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import numpy as np
from PIL import Image
from typing import Dict, Any, List
from src.data.id_generator import IdentityCardGenerator
from src.data.augmentation import IdentityAugmenter
from src.models.document_parser import IDDocumentViTDetector
from src.models.face_verifier import SiameseFaceVerifier
from src.models.forgery_detector import SiameseForgeryNet
from src.inference.engine import MultimodalVerificationEngine
from src.evaluation.metrics import IdentityVerificationMetrics

class IdentityBenchmarkHarness:
    """
    Benchmarks IdentityGuardian AI pipeline under baseline vs calibrated self-healing modes.
    Evaluates layout parsing, biometric face verifications, and forgery screening rates.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generator = IdentityCardGenerator(seed=42)
        self.augmenter = IdentityAugmenter(config)
        
        # Load model parameters dynamically
        model_cfg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "configs", "model_config.yaml"))
        try:
            import yaml
            with open(model_cfg_path, "r") as f:
                model_config = yaml.safe_load(f)
        except Exception:
            model_config = {
                "face_verifier": {"margin": 0.6},
                "forgery_detector": {"tamper_threshold": 0.5}
            }
            
        self.layout_model = IDDocumentViTDetector(num_classes=4, embed_dim=256)
        self.face_model = SiameseFaceVerifier(embedding_dim=128)
        self.forgery_model = SiameseForgeryNet(embedding_dim=128)
        
        self.engine = MultimodalVerificationEngine(
            layout_model=self.layout_model,
            face_model=self.face_model,
            forgery_model=self.forgery_model,
            config=model_config
        )

    def execute_evaluations(self, num_samples: int = 10) -> Dict[str, Any]:
        print(f"Beginning IdentityGuardian AI benchmarks over {num_samples} iterations...")
        logs = {
            "baseline": {"similarity_scores": [], "labels": [], "cer": [], "latency": []},
            "self_healing": {"similarity_scores": [], "labels": [], "cer": [], "latency": []}
        }
        
        # Generate mock selfies
        selfie_img = Image.new("RGB", (128, 128), (250, 210, 180))
        
        for i in range(num_samples):
            # Generate ID
            card_img, face_img, meta = self.generator.generate_id_card()
            
            # Apply distortion + tamper attacks
            distorted_img, augmented_meta, augment_info = self.augmenter.augment_pipeline(card_img, meta)
            
            # Label: 0 for genuine match (same subject, untampered), 1 for mismatched/tampered face swap
            ground_truth_label = 1.0 if (augment_info["is_tampered"] and augment_info["tamper_type"] == "face_swap") else 0.0
            
            # 1. Baseline Run
            start = time.time()
            self.engine.entropy_threshold = 10.0 # disable visual fallbacks
            base_res = self.engine.verify_identity(distorted_img, selfie_image=selfie_img)
            base_latency = time.time() - start
            
            # 2. Self-Healing Run
            start = time.time()
            self.engine.entropy_threshold = 0.50 # enable fallbacks
            healed_res = self.engine.verify_identity(distorted_img, selfie_image=selfie_img)
            healed_latency = time.time() - start
            
            # Record similarities
            logs["baseline"]["similarity_scores"].append(base_res["face_match_score"])
            logs["baseline"]["labels"].append(ground_truth_label)
            logs["baseline"]["latency"].append(base_latency)
            
            logs["self_healing"]["similarity_scores"].append(healed_res["face_match_score"])
            logs["self_healing"]["labels"].append(ground_truth_label)
            logs["self_healing"]["latency"].append(healed_latency)
            
            # Evaluate CER on name text field
            gt_name = augmented_meta["ground_truth"]["name"]
            base_name = base_res["extracted_fields"].get("name", "")
            healed_name = healed_res["extracted_fields"].get("name", "")
            
            logs["baseline"]["cer"].append(IdentityVerificationMetrics.compute_cer(gt_name, base_name))
            logs["self_healing"]["cer"].append(IdentityVerificationMetrics.compute_cer(gt_name, healed_name))

        # Calculate FAR & FRR biometric metrics
        threshold = 0.60
        base_far, base_frr = IdentityVerificationMetrics.compute_far_frr(
            logs["baseline"]["similarity_scores"], logs["baseline"]["labels"], threshold
        )
        healed_far, healed_frr = IdentityVerificationMetrics.compute_far_frr(
            logs["self_healing"]["similarity_scores"], logs["self_healing"]["labels"], threshold
        )
        
        # Calculate ROC-AUC values
        base_auc = IdentityVerificationMetrics.compute_roc_auc(
            logs["baseline"]["similarity_scores"], logs["baseline"]["labels"]
        )
        healed_auc = IdentityVerificationMetrics.compute_roc_auc(
            logs["self_healing"]["similarity_scores"], logs["self_healing"]["labels"]
        )

        print("\n" + "="*50)
        print("         IDENTITYGUARDIAN AI BENCHMARK REPORT      ")
        print("="*50)
        print(f"Metric              | Baseline      | Self-Healing")
        print("-"*50)
        print(f"Face Match AUC      | {base_auc:.4f}        | {healed_auc:.4f}")
        print(f"False Accept (FAR)  | {base_far*100:.1f}%         | {healed_far*100:.1f}%")
        print(f"False Reject (FRR)  | {base_frr*100:.1f}%         | {healed_frr*100:.1f}%")
        print(f"Name Text CER       | {np.mean(logs['baseline']['cer']):.4f}        | {np.mean(logs['self_healing']['cer']):.4f}")
        print(f"Avg Latency (sec)   | {np.mean(logs['baseline']['latency']):.4f}        | {np.mean(logs['self_healing']['latency']):.4f}")
        print("="*50)
        
        return logs

if __name__ == "__main__":
    import yaml
    with open("configs/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    harness = IdentityBenchmarkHarness(data_cfg)
    harness.execute_evaluations(num_samples=5)

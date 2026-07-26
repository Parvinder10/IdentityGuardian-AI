import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from typing import Dict, Any, List
from src.inference.adaptive_engine import AdaptiveVerificationPlanner

class AdaptiveKYCBenchmark:
    """
    Monte Carlo benchmarking harness comparing Static KYC Pipelines
    against the Adaptive Identity Verification Engine.
    Simulates 150 unique verification requests spanning:
    1. Low-Risk Genuine Users
    2. Borderline / Low-Lighting Genuine Users
    3. Sophisticated Spoof / Fraud Attack Vectors
    """
    def __init__(self):
        self.planner = AdaptiveVerificationPlanner()
        self.num_trials = 150

    def generate_simulated_user(self) -> Dict[str, Any]:
        profile_type = np.random.choice(["genuine_clean", "genuine_noisy", "fraud_attack"], p=[0.60, 0.25, 0.15])
        
        if profile_type == "genuine_clean":
            return {
                "profile": "Clean Genuine",
                "face_sim": float(np.random.uniform(0.85, 0.98)),
                "ocr_errors": float(np.random.uniform(0.0, 0.10)),
                "forgery_detected": False,
                "metadata_mismatch": False,
                "is_fraud": False
            }
        elif profile_type == "genuine_noisy":
            return {
                "profile": "Noisy Genuine (Poor Lighting)",
                "face_sim": float(np.random.uniform(0.55, 0.75)),
                "ocr_errors": float(np.random.uniform(0.08, 0.25)),
                "forgery_detected": False,
                "metadata_mismatch": np.random.choice([True, False], p=[0.2, 0.8]),
                "is_fraud": False
            }
        else: # fraud_attack
            return {
                "profile": "Spoof/Tamper Fraud Attack",
                "face_sim": float(np.random.uniform(0.15, 0.48)),
                "ocr_errors": float(np.random.uniform(0.30, 0.80)),
                "forgery_detected": np.random.choice([True, False], p=[0.9, 0.1]),
                "metadata_mismatch": np.random.choice([True, False], p=[0.7, 0.3]),
                "is_fraud": True
            }

    def run_benchmark(self) -> Dict[str, Any]:
        print(f"Starting Adaptive KYC Benchmarking Simulation over {self.num_trials} Monte Carlo trials...")
        
        static_stats = {
            "steps": [],
            "latency": [],
            "friction": [],
            "manual_reviews": 0,
            "false_accepts": 0,
            "false_rejects": 0,
            "approvals": 0,
            "rejections": 0
        }
        
        adaptive_stats = {
            "steps": [],
            "latency": [],
            "friction": [],
            "manual_reviews": 0,
            "false_accepts": 0,
            "false_rejects": 0,
            "approvals": 0,
            "rejections": 0
        }
        
        for _ in range(self.num_trials):
            user = self.generate_simulated_user()
            
            # --- 1. Simulate Static KYC Pipeline (Fixed Steps: ID scan -> Selfie -> Forgery -> Verify) ---
            # All steps always execute. Latency and friction are fixed.
            static_steps = 3.0
            static_latency = 4.8 # fixed processing/wait latency
            static_friction = 1.0 # full user effort
            
            # Calculate metrics
            scores = self.planner.scorer.compute_scores(
                face_sim=user["face_sim"],
                ocr_errors=user["ocr_errors"],
                forgery_detected=user["forgery_detected"],
                metadata_mismatch=user["metadata_mismatch"]
            )
            
            trust = scores["overall_trust"]
            fraud = scores["fraud_probability"]
            
            # Static decision boundary (requires higher strictness due to lack of interactive steps)
            if trust >= 0.82:
                static_decision = "APPROVE"
            elif trust <= 0.45 or fraud >= 0.70:
                static_decision = "REJECT"
            else:
                static_decision = "ESCALATE_MANUAL"
                
            # Log static results
            static_stats["steps"].append(static_steps)
            static_stats["latency"].append(static_latency)
            static_stats["friction"].append(static_friction)
            
            if static_decision == "APPROVE":
                static_stats["approvals"] += 1
                if user["is_fraud"]:
                    static_stats["false_accepts"] += 1
            elif static_decision == "REJECT":
                static_stats["rejections"] += 1
                if not user["is_fraud"]:
                    static_stats["false_rejects"] += 1
            else:
                static_stats["manual_reviews"] += 1

            # --- 2. Simulate Adaptive KYC Pipeline ---
            step = 1
            adaptive_latency = 0.5 # initial ID scan and parser time
            adaptive_friction = 0.2
            
            while step <= self.planner.max_steps:
                # Retrieve planner recommendations based on current subset of information
                # Step 1: only has ID, OCR and Forgery available. Face match is 0.0 until selfie.
                if step == 1:
                    state = {
                        "face_sim": 0.0,
                        "ocr_errors": user["ocr_errors"],
                        "forgery_detected": user["forgery_detected"],
                        "metadata_mismatch": False
                    }
                else:
                    state = {
                        "face_sim": user["face_sim"],
                        "ocr_errors": user["ocr_errors"],
                        "forgery_detected": user["forgery_detected"],
                        "metadata_mismatch": user["metadata_mismatch"]
                    }
                    
                plan = self.planner.plan_next_step(state, step=step)
                next_action = plan["next_action"]
                
                if next_action in ["APPROVE", "REJECT", "ESCALATE_MANUAL"]:
                    adaptive_decision = next_action
                    break
                else:
                    # Increment friction and latency for intermediate steps
                    spec = self.planner.optimizer.action_specs[next_action]
                    adaptive_latency += spec["latency"]
                    adaptive_friction += spec["friction"]
                    step += 1
            else:
                adaptive_decision = "ESCALATE_MANUAL"

            adaptive_stats["steps"].append(step)
            adaptive_stats["latency"].append(adaptive_latency)
            adaptive_stats["friction"].append(adaptive_friction)
            
            if adaptive_decision == "APPROVE":
                adaptive_stats["approvals"] += 1
                if user["is_fraud"]:
                    adaptive_stats["false_accepts"] += 1
            elif adaptive_decision == "REJECT":
                adaptive_stats["rejections"] += 1
                if not user["is_fraud"]:
                    adaptive_stats["false_rejects"] += 1
            else:
                adaptive_stats["manual_reviews"] += 1

        # --- 3. Compute Comparative Research Aggregates ---
        total_genuines = sum(1 for _ in range(self.num_trials) if not self.generate_simulated_user()["is_fraud"])
        total_frauds = self.num_trials - total_genuines
        
        static_far = (static_stats["false_accepts"] / max(1, total_frauds)) * 100
        static_frr = (static_stats["false_rejects"] / max(1, total_genuines)) * 100
        static_manual_rate = (static_stats["manual_reviews"] / self.num_trials) * 100
        
        adaptive_far = (adaptive_stats["false_accepts"] / max(1, total_frauds)) * 100
        adaptive_frr = (adaptive_stats["false_rejects"] / max(1, total_genuines)) * 100
        adaptive_manual_rate = (adaptive_stats["manual_reviews"] / self.num_trials) * 100

        self.print_latex_report(
            static_steps=np.mean(static_stats["steps"]),
            adaptive_steps=np.mean(adaptive_stats["steps"]),
            static_latency=np.mean(static_stats["latency"]),
            adaptive_latency=np.mean(adaptive_stats["latency"]),
            static_friction=np.mean(static_stats["friction"]),
            adaptive_friction=np.mean(adaptive_stats["friction"]),
            static_manual=static_manual_rate,
            adaptive_manual=adaptive_manual_rate,
            static_far=static_far,
            adaptive_far=adaptive_far,
            static_frr=static_frr,
            adaptive_frr=adaptive_frr
        )

    def print_latex_report(self, static_steps: float, adaptive_steps: float,
                           static_latency: float, adaptive_latency: float,
                           static_friction: float, adaptive_friction: float,
                           static_manual: float, adaptive_manual: float,
                           static_far: float, adaptive_far: float,
                           static_frr: float, adaptive_frr: float):
        
        latex_table = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Identity Verification Pipeline Comparison: Static vs. Adaptive Engine}}
\\label{{tab:kyc_comparison}}
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Evaluation Metric}} & \\textbf{{Static KYC Pipeline}} & \\textbf{{Adaptive AI Engine}} \\\\ \\hline
Avg. Verification Steps & {static_steps:.2f} & {adaptive_steps:.2f} \\\\
Avg. Execution Latency & {static_latency:.2f}\\,s & {adaptive_latency:.2f}\\,s \\\\
Customer Friction Index (0-1) & {static_friction:.2f} & {adaptive_friction:.2f} \\\\
Manual Review Escalation Rate & {static_manual:.1f}\\% & {adaptive_manual:.1f}\\% \\\\
False Acceptance Rate (FAR) & {static_far:.2f}\\% & {adaptive_far:.2f}\\% \\\\
False Rejection Rate (FRR) & {static_frr:.2f}\\% & {adaptive_frr:.2f}\\% \\\\ \\hline
\\end{{tabular}}
\\end{{table}}
"""
        print("\n" + "="*80)
        print("          ADAPTIVE VERIFICATION COMPARATIVE RESEARCH REPORT (LaTeX)")
        print("="*80)
        print(latex_table)
        print("="*80 + "\n")


if __name__ == "__main__":
    benchmark = AdaptiveKYCBenchmark()
    benchmark.run_benchmark()

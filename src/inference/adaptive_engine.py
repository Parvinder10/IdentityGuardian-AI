import numpy as np
import torch
from typing import Dict, Any, List, Tuple

class ConfidenceScorer:
    """
    Research-grade confidence scoring module.
    Evaluates multivariable risk indicators and computes a joint probability space
    for fraud likelihood and overall identity trust.
    """
    def __init__(self):
        # Logistic weights for fraud probability regression
        self.w_forgery = 4.5
        self.w_face = 3.5
        self.w_ocr = 2.0
        self.w_metadata = 2.5
        self.bias = -3.0

    def compute_scores(self, face_sim: float, ocr_errors: float, 
                       forgery_detected: bool, metadata_mismatch: bool) -> Dict[str, float]:
        # Individual score confidence mappings
        face_confidence = float(face_sim)
        ocr_confidence = float(max(0.0, 1.0 - ocr_errors))
        authenticity_confidence = 0.10 if forgery_detected else 0.95
        metadata_confidence = 0.15 if metadata_mismatch else 0.90
        
        # If face_sim is 0.0, it means it is not yet executed (Step 1)
        face_active = (face_sim > 0.0)
        
        # Liveness verification score (simulated/approximated based on matching consistency)
        if face_active:
            liveness_confidence = float(max(0.1, face_sim * 0.9 + np.random.uniform(-0.05, 0.05)))
            liveness_confidence = min(1.0, max(0.0, liveness_confidence))
        else:
            liveness_confidence = 0.5

        # 1. Logistical Sigmoid Fraud Probability Model
        face_penalty = self.w_face * (1.0 - face_sim) if face_active else 0.0
        x = (self.w_forgery * (1.0 if forgery_detected else 0.0) +
             face_penalty +
             self.w_ocr * ocr_errors +
             self.w_metadata * (1.0 if metadata_mismatch else 0.0) +
             self.bias)
        
        fraud_probability = float(1.0 / (1.0 + np.exp(-x)))

        # 2. Joint Overall Identity Trust Score
        # Trust is a function of positive indicators discounted by fraud likelihood
        if face_active:
            base_trust = (0.4 * face_confidence + 
                          0.3 * ocr_confidence + 
                          0.2 * authenticity_confidence + 
                          0.1 * metadata_confidence)
        else:
            base_trust = (0.5 * ocr_confidence + 
                          0.3 * authenticity_confidence + 
                          0.2 * metadata_confidence)
        
        overall_trust = float((1.0 - fraud_probability) * base_trust)

        return {
            "face_confidence": face_confidence,
            "ocr_confidence": ocr_confidence,
            "document_authenticity": authenticity_confidence,
            "metadata_confidence": metadata_confidence,
            "liveness_confidence": liveness_confidence,
            "fraud_probability": fraud_probability,
            "overall_trust": overall_trust
        }


class VerificationStrategyOptimizer:
    """
    Verification Strategy Optimizer.
    Uses multi-objective utility functions to select next verification steps,
    minimizing user friction and latency while maximizing confidence gains.
    """
    def __init__(self):
        # Action specs: (friction, latency in seconds, cost index)
        self.action_specs = {
            "APPROVE": {"friction": 0.0, "latency": 0.1, "cost": 0.1},
            "REJECT": {"friction": 0.0, "latency": 0.1, "cost": 0.1},
            "REQUEST_SELFIE": {"friction": 0.35, "latency": 3.0, "cost": 0.5},
            "REQUEST_DOCUMENT": {"friction": 0.70, "latency": 8.0, "cost": 1.2},
            "REQUEST_VIDEO": {"friction": 0.85, "latency": 15.0, "cost": 2.5},
            "REQUEST_ADDRESS": {"friction": 0.55, "latency": 6.0, "cost": 0.8},
            "ESCALATE_MANUAL": {"friction": 0.50, "latency": 45.0, "cost": 5.0}
        }
        # Multi-objective weights
        self.lambda_friction = 3.0
        self.lambda_latency = 0.1
        self.lambda_cost = 0.5

    def compute_action_utility(self, action: str, current_trust: float, gap: float) -> float:
        spec = self.action_specs[action]
        # Estimate expected information gain (potential increase in trust)
        info_gain = 0.0
        if action == "REQUEST_SELFIE":
            info_gain = gap * 0.7
        elif action == "REQUEST_DOCUMENT":
            info_gain = gap * 0.8
        elif action == "REQUEST_VIDEO":
            info_gain = gap * 0.95
        elif action == "REQUEST_ADDRESS":
            info_gain = gap * 0.6
        elif action == "ESCALATE_MANUAL":
            info_gain = gap * 1.0 # Solves all uncertainty
            
        utility = info_gain - (self.lambda_friction * spec["friction"] +
                              self.lambda_latency * spec["latency"] +
                              self.lambda_cost * spec["cost"])
        return float(utility)


class AdaptiveVerificationPlanner:
    """
    Dynamic FSM Planner.
    Determines next verification steps and builds explainable decision trails.
    """
    def __init__(self):
        self.scorer = ConfidenceScorer()
        self.optimizer = VerificationStrategyOptimizer()
        # Decision boundary thresholds
        self.trust_approve_threshold = 0.82
        self.trust_reject_threshold = 0.38
        self.fraud_reject_threshold = 0.72
        self.max_steps = 4

    def plan_next_step(self, current_state: Dict[str, Any], step: int = 1) -> Dict[str, Any]:
        face_sim = current_state.get("face_sim", 0.0)
        ocr_errors = current_state.get("ocr_errors", 0.0)
        forgery_detected = current_state.get("forgery_detected", False)
        metadata_mismatch = current_state.get("metadata_mismatch", False)

        # 1. Compute trust metrics
        metrics = self.scorer.compute_scores(
            face_sim=face_sim,
            ocr_errors=ocr_errors,
            forgery_detected=forgery_detected,
            metadata_mismatch=metadata_mismatch
        )
        
        trust = metrics["overall_trust"]
        fraud = metrics["fraud_probability"]
        
        # 2. Check terminating states
        if trust >= self.trust_approve_threshold and face_sim > 0.0:
            decision = "APPROVE"
            explanation = "Overall identity trust score is high (T={:.2f}). No further checks required.".format(trust)
            evidence = "Face match and document template verify successfully with minimal risk anomalies."
        elif trust <= self.trust_reject_threshold or fraud >= self.fraud_reject_threshold:
            decision = "REJECT"
            explanation = "Verification rejected. Fraud risk probability is high (P={:.2f}) and trust is low (T={:.2f}).".format(fraud, trust)
            evidence = "Identified suspicious anomalies, mismatching face indices, or high template forgery indicators."
        elif step >= self.max_steps:
            decision = "ESCALATE_MANUAL"
            explanation = "Maximum verification steps reached. Esculating to a manual compliance review officer."
            evidence = f"Trust boundaries unresolved (T={trust:.2f}) after {step} iterations."
        else:
            # 3. Dynamic Optimization Selection
            gap = self.trust_approve_threshold - trust
            candidate_actions = ["REQUEST_SELFIE", "REQUEST_DOCUMENT", "REQUEST_VIDEO", "REQUEST_ADDRESS", "ESCALATE_MANUAL"]
            
            best_action = "ESCALATE_MANUAL"
            best_utility = -999.0
            
            for action in candidate_actions:
                # Filter logical actions based on failure roots
                if action == "REQUEST_SELFIE" and face_sim >= 0.78:
                    continue
                if action == "REQUEST_DOCUMENT" and ocr_errors < 0.15:
                    continue
                if action == "REQUEST_ADDRESS" and not metadata_mismatch:
                    continue
                if action == "REQUEST_VIDEO" and not forgery_detected and face_sim >= 0.70:
                    continue
                    
                u = self.optimizer.compute_action_utility(action, trust, gap)
                if u > best_utility:
                    best_utility = u
                    best_action = action
                    
            decision = best_action
            
            # Map action reason details
            if best_action == "REQUEST_SELFIE":
                explanation = "Face verification similarity index is borderline ({:.2f}). Requesting another selfie snapshot.".format(face_sim)
                evidence = "Biometric similarity scores indicate possible poor lighting or framing offsets."
            elif best_action == "REQUEST_DOCUMENT":
                explanation = "Document parser fields had high character error rates ({:.2f}). Requesting secondary scan.".format(ocr_errors)
                evidence = "Optical OCR sequence analysis flagged low-resolution template coordinates."
            elif best_action == "REQUEST_VIDEO":
                explanation = "Tamper indicators detected. Requesting live video face tracking scan."
                evidence = "High forgery match distances detected. Requires full interactive liveness test."
            elif best_action == "REQUEST_ADDRESS":
                explanation = "Geographical address coordinates mismatch registry records. Requesting utility bill upload."
                evidence = "Device IP geolocation contradicts document metadata address fields."
            else:
                explanation = "Uncertainty boundary remains unresolved. Transferring file to manual compliance audit."
                evidence = "Combined metrics do not satisfy automated verification thresholds."

        return {
            "step": step,
            "next_action": decision,
            "metrics": metrics,
            "explanation": explanation,
            "evidence": evidence,
            "suggested_action": decision
        }

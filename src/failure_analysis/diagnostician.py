import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any

class IdentityFailureDiagnostician:
    """
    Groups KYC verification failures and isolates biometrics/OCR flaws.
    Uses K-Means to cluster failures based on lighting, card tilt, and similarity scores.
    """
    def __init__(self, n_clusters: int = 3):
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.explanations = {
            0: "Failed due to severe illumination mismatch. Lighting variations between ID card photo and webcam selfie crop.",
            1: "Failed due to document perspective/rotation skew. The layout parser failed to isolate profile picture region accurately.",
            2: "Failed due to high probability biometric spoof. Digital alteration/face-swap warning flags triggered."
        }

    def _extract_kyc_features(self, logs: List[Dict[str, Any]]) -> np.ndarray:
        features = []
        for log in logs:
            features.append([
                log.get("illumination_shift", 0.0),
                log.get("rotation_tilt_deg", 0.0),
                log.get("face_match_similarity", 0.0),
                float(log.get("forgery_detected", 0))
            ])
        return np.array(features)

    def diagnose_kyc_failures(self, failure_logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not failure_logs:
            return []
            
        features = self._extract_kyc_features(failure_logs)
        self.kmeans.fit(features)
        labels = self.kmeans.labels_
        
        diagnosed = []
        for idx, log in enumerate(failure_logs):
            cluster_id = int(labels[idx])
            explanation = self.explanations.get(cluster_id, "Unknown KYC verification failure.")
            
            analyzed = log.copy()
            analyzed["failure_cluster"] = cluster_id
            analyzed["root_cause_explanation"] = explanation
            diagnosed.append(analyzed)
            
        return diagnosed

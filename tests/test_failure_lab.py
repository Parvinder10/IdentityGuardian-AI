import pytest
from src.models.failure_lab import FailureClassifier, ExplainabilityEngine

def test_failure_classification_healthy():
    classifier = FailureClassifier()
    res = classifier.classify_failure(
        face_sim=0.92,
        ocr_errors=0.02,
        forgery_detected=False,
        metadata_mismatch=False,
        rag_score=0.88,
        hallucination_detected=False
    )
    
    assert "None (Healthy Scan)" in res["classified_failures"]
    assert "Minor environmental signal attenuation" in res["failure_cluster"]


def test_failure_classification_anomalies():
    classifier = FailureClassifier()
    
    # 1. OCR + Biometric Failures under low contrast cluster
    res_drift = classifier.classify_failure(
        face_sim=0.62,
        ocr_errors=0.28,
        forgery_detected=False,
        metadata_mismatch=False,
        rag_score=0.90,
        hallucination_detected=False
    )
    assert "OCR Failure" in res_drift["classified_failures"]
    assert "Face Verification Failure" in res_drift["classified_failures"]
    assert "Low-contrast lighting biometrics drift" in res_drift["failure_cluster"]
    
    # 2. Forgery attack
    res_spoof = classifier.classify_failure(
        face_sim=0.95,
        ocr_errors=0.01,
        forgery_detected=True,
        metadata_mismatch=False,
        rag_score=0.90,
        hallucination_detected=False
    )
    assert "Forgery Detection Failure" in res_spoof["classified_failures"]
    assert "Advanced physical/presentation attack vectors" in res_spoof["failure_cluster"]


def test_explainability_engine_payload():
    engine = ExplainabilityEngine()
    res = engine.generate_all_explanations()
    
    required_keys = ["integrated_gradients", "gradcam", "shap", "lime", "attention"]
    for key in required_keys:
        assert key in res
        assert res[key].startswith("data:image/svg+xml;base64,")

import pytest
from src.inference.adaptive_engine import ConfidenceScorer, AdaptiveVerificationPlanner, VerificationStrategyOptimizer

def test_confidence_scorer_logic():
    scorer = ConfidenceScorer()
    
    # Test high trust scenario
    scores_high = scorer.compute_scores(
        face_sim=0.92,
        ocr_errors=0.0,
        forgery_detected=False,
        metadata_mismatch=False
    )
    assert scores_high["overall_trust"] > 0.80
    assert scores_high["fraud_probability"] < 0.20
    
    # Test forgery threat scenario
    scores_forgery = scorer.compute_scores(
        face_sim=0.50,
        ocr_errors=0.3,
        forgery_detected=True,
        metadata_mismatch=False
    )
    assert scores_forgery["fraud_probability"] > 0.70
    assert scores_forgery["overall_trust"] < 0.35


def test_adaptive_planner_fsm():
    planner = AdaptiveVerificationPlanner()
    
    # 1. High Trust approval trigger
    state_ok = {"face_sim": 0.95, "ocr_errors": 0.0, "forgery_detected": False, "metadata_mismatch": False}
    plan_ok = planner.plan_next_step(state_ok, step=1)
    assert plan_ok["next_action"] == "APPROVE"
    assert "high" in plan_ok["explanation"].lower()
    
    # 2. Low Trust rejection trigger
    state_bad = {"face_sim": 0.25, "ocr_errors": 0.6, "forgery_detected": True, "metadata_mismatch": True}
    plan_bad = planner.plan_next_step(state_bad, step=1)
    assert plan_bad["next_action"] == "REJECT"
    assert "reject" in plan_bad["explanation"].lower()
    
    # 3. Intermediate border-line routing (low face similarity -> request selfie)
    state_selfie = {"face_sim": 0.62, "ocr_errors": 0.0, "forgery_detected": False, "metadata_mismatch": False}
    plan_selfie = planner.plan_next_step(state_selfie, step=1)
    assert plan_selfie["next_action"] == "REQUEST_SELFIE"
    
    # 4. Max steps escalation route
    state_limit = {"face_sim": 0.65, "ocr_errors": 0.2, "forgery_detected": False, "metadata_mismatch": False}
    plan_limit = planner.plan_next_step(state_limit, step=4)
    assert plan_limit["next_action"] == "ESCALATE_MANUAL"


def test_strategy_utility_optimizations():
    optimizer = VerificationStrategyOptimizer()
    
    # Verify that requesting a selfie under borderline face match is highly optimal
    u_selfie = optimizer.compute_action_utility("REQUEST_SELFIE", current_trust=0.60, gap=0.25)
    u_doc = optimizer.compute_action_utility("REQUEST_DOCUMENT", current_trust=0.60, gap=0.25)
    
    # Doc request has higher user friction and latency costs, making Selfie more utility-optimal
    assert u_selfie > u_doc

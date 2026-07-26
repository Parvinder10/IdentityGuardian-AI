import pytest
from src.inference.graph_engine import EvidenceGraphManager
from src.inference.graph_agents import CoordinatorAgent

def test_evidence_graph_insertions():
    manager = EvidenceGraphManager()
    
    # Add nodes
    manager.add_node("test_user_1", "User", {"label": "User One", "status": "PENDING"})
    manager.add_node("test_device_1", "Device", {"label": "Device One"})
    
    # Add connection
    manager.add_edge("test_user_1", "test_device_1", "SAME_DEVICE")
    
    assert "test_user_1" in manager.nx_graph
    assert "test_device_1" in manager.nx_graph
    assert manager.nx_graph.has_edge("test_user_1", "test_device_1", "SAME_DEVICE")


def test_fraud_ring_clustering_detection():
    manager = EvidenceGraphManager()
    # Reset default graph for testing clean components
    manager.nx_graph.clear()
    
    # Add User 1 and User 2 linked to the same face
    manager.add_node("user_1", "User", {"node_type": "User"})
    manager.add_node("user_2", "User", {"node_type": "User"})
    manager.add_node("shared_face_1", "Face", {"node_type": "Face"})
    
    manager.add_edge("user_1", "shared_face_1", "SAME_FACE")
    manager.add_edge("user_2", "shared_face_1", "SAME_FACE")
    
    rings = manager.detect_fraud_rings()
    assert len(rings) == 1
    assert "user_1" in rings[0]
    assert "user_2" in rings[0]


def test_multi_agent_investigation_aggregation():
    manager = EvidenceGraphManager()
    coordinator = CoordinatorAgent()
    
    # Run audit on a borderline file
    report = coordinator.investigate_user(
        user_id="user_bob",
        graph_manager=manager,
        face_sim=0.55,
        ocr_errors=0.10,
        forgery_detected=False,
        name="Bob Johnson"
    )
    
    assert report["verdict"] in ["REJECT", "AUDIT"]
    assert "Multi-Agent KYC Investigation Report" in report["report_markdown"]
    assert report["ocr_analysis"]["risk_rating"] == "LOW"
    assert report["face_analysis"]["risk_rating"] == "HIGH" # due to face_sim = 0.55 < 0.60
    assert report["fraud_analysis"]["risk_rating"] == "CRITICAL" # due to shared biometric face template in bootstrap

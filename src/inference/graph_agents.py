from typing import Dict, Any, List
from src.inference.graph_engine import EvidenceGraphManager

class OCRAgent:
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        ocr_errors = context.get("ocr_errors", 0.0)
        name = context.get("name", "Unknown")
        
        # Analyze name spelling structures and missing layout fields
        confidence = max(0.0, 1.0 - ocr_errors)
        risk = "LOW"
        notes = "Extracted name validates layout template requirements."
        
        if ocr_errors > 0.3:
            risk = "HIGH"
            notes = f"High characters error rates ({ocr_errors:.2f}). Sequence layout extraction is borderline."
        elif name == "Jane Doe" or name == "Unknown":
            risk = "MEDIUM"
            notes = "Default template name fallback detected. Optical parsing failed to extract custom identity name."
            
        return {
            "agent": "OCR Agent",
            "confidence": float(confidence),
            "risk_rating": risk,
            "findings": notes
        }


class FaceAgent:
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        face_sim = context.get("face_sim", 0.0)
        
        risk = "LOW"
        notes = f"Biometric matching verifies user identity (Similarity: {face_sim:.4f})."
        
        if face_sim == 0.0:
            risk = "HIGH"
            notes = "Biometric face verification has not been performed yet. Awaiting live webcam selfie."
        elif face_sim < 0.60:
            risk = "HIGH"
            notes = f"Critical facial match warning! Biometric similarity ({face_sim:.4f}) is below security margins."
        elif face_sim < 0.78:
            risk = "MEDIUM"
            notes = f"Borderline biometric score ({face_sim:.4f}). Potential changes in lighting, angles, or frame cropping."
            
        return {
            "agent": "Face Agent",
            "confidence": float(face_sim) if face_sim > 0.0 else 0.1,
            "risk_rating": risk,
            "findings": notes
        }


class DocumentAgent:
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        forgery_detected = context.get("forgery_detected", False)
        
        risk = "LOW"
        confidence = 0.95
        notes = "Document template checks verify structural authenticity."
        
        if forgery_detected:
            risk = "CRITICAL"
            confidence = 0.10
            notes = "Tamper alert! Layout pattern networks detect visual splicing, edge inconsistencies, or pixel frequency anomalies."
            
        return {
            "agent": "Document Agent",
            "confidence": confidence,
            "risk_rating": risk,
            "findings": notes
        }


class FraudAgent:
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        subgraph = context.get("subgraph", [])
        user_id = context.get("user_id", "")
        
        # Analyze device reuse, IP sharing, and duplicate face templates
        risk = "LOW"
        confidence = 0.95
        shared_devices = []
        shared_ips = []
        shared_faces = []
        
        for item in subgraph:
            if "relation" in item:
                rel = item["relation"]
                src = item["source"]
                tgt = item["target"]
                
                if rel == "SAME_DEVICE" and src != user_id:
                    shared_devices.append(src)
                elif rel == "SAME_NETWORK" and src != user_id:
                    shared_ips.append(src)
                elif rel == "SAME_FACE" and src != user_id:
                    shared_faces.append(src)

        findings = "No suspicious network, device sharing, or face duplication detected."
        
        if shared_faces:
            risk = "CRITICAL"
            confidence = 0.05
            findings = f"FACIAL CLONING WARNING: Face template duplicated across user files: {shared_faces}."
        elif len(shared_devices) >= 2:
            risk = "HIGH"
            confidence = 0.20
            findings = f"DEVICE SPAMMING: Device is shared among multiple profiles: {shared_devices}."
        elif shared_devices or shared_ips:
            risk = "MEDIUM"
            confidence = 0.60
            findings = f"Shared assets: Device linked to {shared_devices}, network IP shared with {shared_ips}."
            
        return {
            "agent": "Fraud Agent",
            "confidence": confidence,
            "risk_rating": risk,
            "findings": findings
        }


class ComplianceAgent:
    def analyze(self, context: Dict[str, Any]) -> Dict[str, Any]:
        face_sim = context.get("face_sim", 0.0)
        ocr_errors = context.get("ocr_errors", 0.0)
        forgery_detected = context.get("forgery_detected", False)
        
        # Compliance requires verified OCR, authentic document, and strong face match
        risk = "LOW"
        notes = "Verification protocol fulfills regulatory KYC compliance mandates."
        
        if forgery_detected:
            risk = "HIGH"
            notes = "KYC protocol rejected due to failed template authenticity checks."
        elif face_sim < 0.60:
            risk = "HIGH"
            notes = "KYC protocol rejected due to insufficient biometric matching score."
        elif face_sim == 0.0:
            risk = "MEDIUM"
            notes = "Verification is incomplete. Live selfie capture is pending."
            
        return {
            "agent": "Compliance Agent",
            "confidence": 0.90,
            "risk_rating": risk,
            "findings": notes
        }


class CoordinatorAgent:
    """
    Coordinator Agent.
    Executes GraphRAG traversal, coordinates sub-agent analysis,
    and compiles a unified natural language investigation report.
    """
    def __init__(self):
        self.ocr_agent = OCRAgent()
        self.face_agent = FaceAgent()
        self.doc_agent = DocumentAgent()
        self.fraud_agent = FraudAgent()
        self.comp_agent = ComplianceAgent()

    def investigate_user(self, user_id: str, graph_manager: EvidenceGraphManager, 
                         face_sim: float = 0.0, ocr_errors: float = 0.0, 
                         forgery_detected: bool = False, name: str = "Unknown") -> Dict[str, Any]:
        
        # 1. GraphRAG context retrieval
        subgraph = graph_manager.traverse_subgraph(user_id, max_depth=2)
        
        context = {
            "user_id": user_id,
            "subgraph": subgraph,
            "face_sim": face_sim,
            "ocr_errors": ocr_errors,
            "forgery_detected": forgery_detected,
            "name": name
        }
        
        # 2. Query specialized agents
        ocr_res = self.ocr_agent.analyze(context)
        face_res = self.face_agent.analyze(context)
        doc_res = self.doc_agent.analyze(context)
        fraud_res = self.fraud_agent.analyze(context)
        comp_res = self.comp_agent.analyze(context)
        
        # 3. Determine overall verdict
        risks = [ocr_res["risk_rating"], face_res["risk_rating"], doc_res["risk_rating"], fraud_res["risk_rating"], comp_res["risk_rating"]]
        
        if "CRITICAL" in risks:
            verdict = "REJECT"
            color = "crimson"
        elif risks.count("HIGH") >= 2:
            verdict = "REJECT"
            color = "crimson"
        elif "HIGH" in risks or "MEDIUM" in risks:
            verdict = "AUDIT"
            color = "orange"
        else:
            verdict = "APPROVE"
            color = "green"
            
        # 4. Compile natural language investigation report (Markdown)
        report = f"""### 🛡️ Multi-Agent KYC Investigation Report: `{user_id}`

#### 1. Executive Summary
The Coordinator Agent synthesized reviews from 5 specialized compliance agents. The overall system verdict is **{verdict}** due to risk profiles identified in the connected evidence graph.

---

#### 2. Connected Evidence Graph (GraphRAG Context)
We traversed the user entity's relationship map up to 2 degrees of separation:
- Total linked nodes/edges found: **{len(subgraph)}**
- **Anomalies Checked**:
  - *Fraud Ring Affiliation*: {fraud_res['findings']}
  - *Identified Name*: `{name}` (OCR confidence: {(ocr_res['confidence']*100):.1f}%)

---

#### 3. Compliance Agent Review Breakdown
- **{ocr_res['agent']}** | Risk: `{ocr_res['risk_rating']}` | Conf: `{ocr_res['confidence']:.2f}`
  - *Findings*: {ocr_res['findings']}
- **{face_res['agent']}** | Risk: `{face_res['risk_rating']}` | Conf: `{face_res['confidence']:.2f}`
  - *Findings*: {face_res['findings']}
- **{doc_res['agent']}** | Risk: `{doc_res['risk_rating']}` | Conf: `{doc_res['confidence']:.2f}`
  - *Findings*: {doc_res['findings']}
- **{fraud_res['agent']}** | Risk: `{fraud_res['risk_rating']}` | Conf: `{fraud_res['confidence']:.2f}`
  - *Findings*: {fraud_res['findings']}
- **{comp_res['agent']}** | Risk: `{comp_res['risk_rating']}` | Conf: `{comp_res['confidence']:.2f}`
  - *Findings*: {comp_res['findings']}

---

#### 4. Coordinator Final Verdict & Recommendation
- **Verdict**: <span style="color: {color}; font-weight: bold; text-transform: uppercase;">{verdict}</span>
- **Suggested Action**: { "Reject user file immediately and lock related device coordinates." if verdict == 'REJECT' else "Route file to manual compliance queue for secondary verification." if verdict == 'AUDIT' else "Approve file and issue credentials." }
"""

        return {
            "verdict": verdict,
            "report_markdown": report,
            "ocr_analysis": ocr_res,
            "face_analysis": face_res,
            "document_analysis": doc_res,
            "fraud_analysis": fraud_res,
            "compliance_analysis": comp_res
        }

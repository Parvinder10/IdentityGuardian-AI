import numpy as np
import torch
from typing import List, Dict, Any, Tuple

class EvaluationMetrics:
    """
    Computes rigorous industrial and research metrics for Document AI.
    Includes Levenshtein-based CER/WER, Layout mAP, and IoU.
    """

    @staticmethod
    def compute_levenshtein_distance(seq1: str, seq2: str) -> int:
        """Computes edit distance between two strings using dynamic programming."""
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1), dtype=int)
        
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j
            
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i, j] = dp[i - 1, j - 1]
                else:
                    dp[i, j] = 1 + min(
                        dp[i - 1, j],    # Deletion
                        dp[i, j - 1],    # Insertion
                        dp[i - 1, j - 1] # Substitution
                    )
        return dp[m, n]

    @classmethod
    def compute_cer(cls, reference: str, hypothesis: str) -> float:
        """Computes Character Error Rate (CER)."""
        ref_len = len(reference)
        if ref_len == 0:
            return 1.0 if len(hypothesis) > 0 else 0.0
        dist = cls.compute_levenshtein_distance(reference, hypothesis)
        return min(1.0, dist / ref_len)

    @classmethod
    def compute_wer(cls, reference: str, hypothesis: str) -> float:
        """Computes Word Error Rate (WER)."""
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        ref_len = len(ref_words)
        if ref_len == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
            
        word_to_char = {}
        char_idx = 33
        
        def map_words_to_chars(words):
            nonlocal char_idx
            res = ""
            for w in words:
                if w not in word_to_char:
                    word_to_char[w] = chr(char_idx)
                    char_idx += 1
                res += word_to_char[w]
            return res
            
        ref_char_seq = map_words_to_chars(ref_words)
        hyp_char_seq = map_words_to_chars(hyp_words)
        
        dist = cls.compute_levenshtein_distance(ref_char_seq, hyp_char_seq)
        return min(1.0, dist / ref_len)

    @staticmethod
    def compute_iou(box_a: List[float], box_b: List[float]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes [ymin, xmin, ymax, xmax]."""
        y1 = max(box_a[0], box_b[0])
        x1 = max(box_a[1], box_b[1])
        y2 = min(box_a[2], box_b[2])
        x2 = min(box_a[3], box_b[3])
        
        inter_area = max(0.0, y2 - y1) * max(0.0, x2 - x1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area
        
        if union_area == 0.0:
            return 0.0
        return inter_area / union_area

    @classmethod
    def compute_map(cls, pred_boxes: List[List[float]], pred_classes: List[int], pred_scores: List[float],
                    gt_boxes: List[List[float]], gt_classes: List[int], iou_threshold: float = 0.5) -> float:
        """Computes Mean Average Precision (mAP) at specified IoU threshold for layout parsing."""
        unique_classes = set(gt_classes)
        if not unique_classes:
            return 0.0
            
        average_precisions = []
        for c in unique_classes:
            c_gt_boxes = [gt_boxes[i] for i, cls_id in enumerate(gt_classes) if cls_id == c]
            c_preds = [(pred_boxes[i], pred_scores[i]) for i, cls_id in enumerate(pred_classes) if cls_id == c]
            c_preds = sorted(c_preds, key=lambda x: x[1], reverse=True)
            
            if not c_gt_boxes:
                continue
            if not c_preds:
                average_precisions.append(0.0)
                continue
                
            tp = np.zeros(len(c_preds))
            fp = np.zeros(len(c_preds))
            gt_matched = np.zeros(len(c_gt_boxes))
            
            for idx, (p_box, _) in enumerate(c_preds):
                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx, gt_box in enumerate(c_gt_boxes):
                    iou = cls.compute_iou(p_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        
                if best_iou >= iou_threshold:
                    if gt_matched[best_gt_idx] == 0:
                        tp[idx] = 1
                        gt_matched[best_gt_idx] = 1
                    else:
                        fp[idx] = 1
                else:
                    fp[idx] = 1
                    
            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / len(c_gt_boxes)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
            
            ap = 0.0
            for r in np.linspace(0.0, 1.0, 11):
                p_at_r = precisions[recalls >= r]
                ap += np.max(p_at_r) / 11.0 if p_at_r.size > 0 else 0.0
            average_precisions.append(ap)
            
        return float(np.mean(average_precisions)) if average_precisions else 0.0


class IdentityVerificationMetrics(EvaluationMetrics):
    """
    Extends base evaluation metrics with biometrics validation scores:
    False Accept Rate (FAR), False Reject Rate (FRR), and ROC-AUC metrics.
    """
    
    @staticmethod
    def compute_far_frr(similarity_scores: List[float], labels: List[float], threshold: float) -> Tuple[float, float]:
        """
        Computes biometric metrics for a given similarity threshold.
        Labels: 0 for genuine match, 1 for fraud/different face swap.
        """
        scores = np.array(similarity_scores)
        y_true = np.array(labels)
        
        # Accepted as genuine if similarity >= threshold
        accepted_mask = (scores >= threshold)
        
        # FAR = False Accepts / Total Negatives (mismatches)
        negatives = (y_true == 1)
        false_accepts = np.sum(accepted_mask & negatives)
        total_negatives = np.sum(negatives)
        far = false_accepts / total_negatives if total_negatives > 0 else 0.0
        
        # FRR = False Rejects / Total Positives (genuines)
        positives = (y_true == 0)
        rejected_mask = (scores < threshold)
        false_rejects = np.sum(rejected_mask & positives)
        total_positives = np.sum(positives)
        frr = false_rejects / total_positives if total_positives > 0 else 0.0
        
        return float(far), float(frr)

    @classmethod
    def compute_roc_auc(cls, similarity_scores: List[float], labels: List[float]) -> float:
        """Approximates Area Under the ROC Curve (ROC-AUC) for verification thresholds."""
        scores = np.array(similarity_scores)
        y_true = np.array(labels)
        
        binary_labels = (y_true == 0).astype(int)
        desc_score_indices = np.argsort(scores)[::-1]
        sorted_scores = scores[desc_score_indices]
        sorted_labels = binary_labels[desc_score_indices]
        
        tp = np.cumsum(sorted_labels)
        fp = np.cumsum(1 - sorted_labels)
        
        tpr = tp / tp[-1] if tp[-1] > 0 else np.zeros_like(tp)
        fpr = fp / fp[-1] if fp[-1] > 0 else np.zeros_like(fp)
        
        auc = 0.0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2.0
        return float(auc)

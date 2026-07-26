import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Tuple

class UncertaintyEstimator:
    """
    Computes statistical and model-based uncertainty.
    Implements Softmax Temperature Scaling, Shannon Entropy, and Split Conformal Prediction.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.conformal_threshold = 0.95 # default coverage alpha=0.05

    def calculate_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Computes Shannon Entropy for probability distribution:
        H(p) = - sum_j (p_j * log(p_j))
        Input shape: [Batch, Classes] or [Batch, SeqLen, Classes]
        """
        # Add epsilon to prevent log(0)
        eps = 1e-9
        entropy = -torch.sum(probabilities * torch.log(probabilities + eps), dim=-1)
        return entropy

    def calibrate_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Applies temperature scaling to raw model logits."""
        return logits / self.temperature

    def fit_temperature_scaling(self, val_logits: torch.Tensor, val_labels: torch.Tensor) -> float:
        """
        Optimizes the temperature T parameter on validation set logits using L-BFGS.
        Minimizes Negative Log-Likelihood (NLL).
        """
        temp_param = nn.Parameter(torch.ones(1) * self.temperature)
        optimizer = optim.LBFGS([temp_param], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = val_logits / temp_param
            loss = nn.CrossEntropyLoss()(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.temperature = float(temp_param.item())
        print(f"Optimal Temperature Calibrated: T = {self.temperature:.4f}")
        return self.temperature

    def compute_ece(self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10) -> float:
        """
        Computes Expected Calibration Error (ECE):
        ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|
        """
        softmaxes = torch.softmax(logits, dim=-1)
        confidences, predictions = torch.max(softmaxes, dim=-1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Mask elements in the bin
            in_bin = confidences.gt(bin_lower.item()) & confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return float(ece.item())

    def calibrate_conformal_prediction(self, cal_scores: List[float], alpha: float = 0.05) -> float:
        """
        Calibrates Split Conformal Prediction threshold.
        Inputs:
            cal_scores: List of conformity scores (e.g. 1 - P(correct_class) on validation set).
            alpha: Significance level (desired coverage is 1 - alpha).
        Returns:
            The threshold score q_hat.
        """
        n = len(cal_scores)
        if n == 0:
            return 1.0
        # Sort conformity scores
        sorted_scores = np.sort(cal_scores)
        # Find index matching quantile (1 - alpha) * (1 + 1/n)
        q_idx = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
        q_idx = min(max(0, q_idx), n - 1)
        
        self.conformal_threshold = sorted_scores[q_idx]
        print(f"Conformal Prediction calibrated. q_hat threshold = {self.conformal_threshold:.4f} for {100*(1-alpha)}% coverage.")
        return self.conformal_threshold

    def get_conformal_prediction_set(self, probabilities: torch.Tensor) -> List[List[int]]:
        """
        Returns the set of classes for each prediction instance that satisfy conformal bounds.
        Inputs:
            probabilities: Model probabilities shape [Batch, Classes].
        Returns:
            List of lists of class indices in the prediction set.
        """
        # Conformity score for each potential class is 1 - prob
        conformity_scores = 1.0 - probabilities
        
        prediction_sets = []
        for i in range(probabilities.shape[0]):
            valid_classes = []
            for cls_idx in range(probabilities.shape[1]):
                # If conformity score <= threshold, include in set
                if conformity_scores[i, cls_idx] <= self.conformal_threshold:
                    valid_classes.append(cls_idx)
            # Ensure prediction set is never empty by fallback to argmax
            if not valid_classes:
                valid_classes.append(int(torch.argmax(probabilities[i]).item()))
            prediction_sets.append(valid_classes)
            
        return prediction_sets

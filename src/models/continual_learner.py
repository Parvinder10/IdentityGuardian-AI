import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple

class ContinualLearnerManager:
    """
    Manages Active Learning sample selection and Elastic Weight Consolidation (EWC)
    regularization to support self-evolving identity verification.
    """
    def __init__(self, model: nn.Module, ewc_lambda: float = 400.0):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_matrix = {}
        self.optimal_weights = {}
        
    def select_active_learning_samples(self, 
                                       face_cosine_similarities: torch.Tensor, 
                                       margin: float = 0.6, 
                                       window: float = 0.15) -> torch.Tensor:
        """
        Active Learning Boundary Sampling:
        Flags samples where face match confidence is borderline (margin - window <= sim <= margin + window).
        These represent highest learning value instances to route to verification pools.
        """
        lower_bound = margin - window
        upper_bound = margin + window
        
        # Binary mask indicating which samples fall in the uncertain decision region
        is_uncertain = (face_cosine_similarities >= lower_bound) & (face_cosine_similarities <= upper_bound)
        return is_uncertain

    def compute_fisher_information(self, dataloader: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Computes diagonal Fisher Information Matrix coefficients:
        F_j = 1/N * sum_i (d L(theta)/d theta_j)^2
        Saves optimal parameter checkpoints.
        """
        self.model.eval()
        fisher = {}
        optimal = {}
        
        # Initialize Fisher metrics dictionary
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher[name] = torch.zeros_like(param.data)
                optimal[name] = param.data.clone()

        # Compute gradient squared expectation
        n_samples = 0
        criterion = nn.CosineEmbeddingLoss()
        
        # Accumulate gradients
        for img_a, img_b, labels in dataloader:
            self.model.zero_grad()
            emb_a = self.model.forward_once(img_a)
            emb_b = self.model.forward_once(img_b)
            
            # Simulated target labels
            target = torch.where(labels == 0.0, 1.0, -1.0)
            loss = criterion(emb_a, emb_b, target)
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += torch.pow(param.grad.data, 2)
            
            n_samples += img_a.size(0)

        # Normalize Fisher values
        for name in fisher:
            fisher[name] = fisher[name] / n_samples
            
        self.fisher_matrix = fisher
        self.optimal_weights = optimal
        return fisher

    def compute_ewc_loss_penalty(self) -> torch.Tensor:
        """
        Calculates the EWC quadratic loss penalty:
        L_EWC = sum_j (lambda/2 * F_j * (theta_j - theta*_j)^2)
        """
        if not self.fisher_matrix or not self.optimal_weights:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
            
        ewc_loss = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_matrix:
                fisher_val = self.fisher_matrix[name]
                opt_weight = self.optimal_weights[name]
                
                # Apply quadratic penalty
                ewc_loss += torch.sum(fisher_val * torch.pow(param - opt_weight, 2))
                
        return (self.ewc_lambda / 2.0) * ewc_loss
        
    def step_continual_training(self, 
                                 batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                 optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """Runs a single continual learning training step with EWC constraints."""
        self.model.train()
        img_a, img_b, labels = batch
        
        emb_a = self.model.forward_once(img_a)
        emb_b = self.model.forward_once(img_b)
        
        target = torch.where(labels == 0.0, 1.0, -1.0)
        base_loss = nn.CosineEmbeddingLoss()(emb_a, emb_b, target)
        
        # Add EWC penalty to base loss to preserve weights
        ewc_penalty = self.compute_ewc_loss_penalty()
        total_loss = base_loss + ewc_penalty
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss

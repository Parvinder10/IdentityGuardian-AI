import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from PIL import Image

class IdentityExplainabilityVisualizer:
    """
    Computes Integrated Gradients for document layout parsing
    and face matching verification decision boundaries.
    """
    def __init__(self):
        pass

    def compute_face_match_attribution(self,
                                      face_model: torch.nn.Module,
                                      img_id_t: torch.Tensor,
                                      img_selfie_t: torch.Tensor,
                                      steps: int = 15,
                                      device: str = "cpu") -> np.ndarray:
        """
        Computes Integrated Gradients w.r.t. face match similarity.
        Highlights which pixels on the ID photo drove verification.
        """
        face_model.eval()
        face_model.to(device)
        
        img_id_t = img_id_t.to(device).clone().detach().requires_grad_(True)
        img_selfie_t = img_selfie_t.to(device).clone().detach()
        
        baseline = torch.zeros_like(img_id_t, device=device)
        alphas = torch.linspace(0, 1, steps, device=device)
        
        grads_sum = torch.zeros_like(img_id_t, device=device)
        
        for alpha in alphas:
            interpolated = baseline + alpha * (img_id_t - baseline)
            interpolated = interpolated.clone().detach().requires_grad_(True)
            
            # Forward pass: get cosine similarity score
            emb_id = face_model.forward_once(interpolated)
            with torch.no_grad():
                emb_selfie = face_model.forward_once(img_selfie_t)
                
            # Cosine similarity metric: dot product of normalized embeddings
            sim_score = torch.sum(emb_id * emb_selfie, dim=-1)
            
            face_model.zero_grad()
            sim_score.backward()
            
            grads_sum += interpolated.grad

        avg_grads = grads_sum / steps
        integrated_grad = (img_id_t - baseline) * avg_grads
        
        ig_np = integrated_grad.squeeze(0).detach().cpu().numpy()
        ig_np = np.sum(np.abs(ig_np), axis=0) # Aggregate absolute gradient across RGB
        
        if ig_np.max() != ig_np.min():
            ig_np = (ig_np - ig_np.min()) / (ig_np.max() - ig_np.min())
            
        return ig_np

    def plot_attribution_overlay(self, face_image: Image.Image, attribution_map: np.ndarray, save_path: str):
        """Overlays the face-match pixel attribution heatmap onto the face crop."""
        width, height = face_image.size
        attrib_resized = np.array(Image.fromarray(attribution_map).resize((width, height), Image.Resampling.BILINEAR))
        
        plt.figure(figsize=(5, 5))
        plt.imshow(face_image)
        plt.imshow(attrib_resized, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.title("Biometrics Pixel Attribution (IG)")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Biometrics explainability map saved at: {save_path}")

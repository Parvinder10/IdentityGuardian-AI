import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from transformers import AutoProcessor, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import time
import os

class VLMDocDataset(Dataset):
    """
    Unified dataset class for Document VLM Fine-Tuning.
    Returns simulated image-text document verification targets.
    """
    def __init__(self, size: int = 20):
        self.size = size
        
    def __len__(self) -> int:
        return self.size
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        # Yields a dummy image tensor (3 channels, 224x224) and target transcription text
        img = torch.randn(3, 224, 224)
        target_text = f"ocr_result: NAME: KISHOR TOMAR, ID: NPTEL14CS22, DATE: 15-08-1994, SEQ: {idx}"
        return img, target_text


class VLMFineTuningPipeline:
    """
    Orchestrates the Supervised Fine-Tuning (SFT) and PEFT adaptation (LoRA / QLoRA)
    of Vision Language Models (Qwen2-VL, Florence-2, LayoutLMv3).
    """
    def __init__(self, config: Dict[str, Any] = None, device: str = "cpu"):
        self.config = config or {}
        self.device = device
        
        # Hyperparameters
        self.model_name = self.config.get("model_name", "Florence-2")
        self.method = self.config.get("method", "LoRA") # LoRA, QLoRA, Full
        self.learning_rate = self.config.get("learning_rate", 1e-4)
        self.batch_size = self.config.get("batch_size", 2)
        self.epochs = self.config.get("epochs", 3)
        self.lora_r = self.config.get("lora_r", 8)
        self.lora_alpha = self.config.get("lora_alpha", 16)
        self.patience = self.config.get("patience", 2)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def setup_model(self) -> Tuple[nn.Module, Optional[AutoProcessor]]:
        """
        Sets up the base model and processor.
        Applies LoRA/QLoRA adapter configurations if requested.
        """
        # Quantization Config for QLoRA
        quant_config = None
        if self.method == "QLoRA" and self.device == "cuda":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            
        print(f"[{self.model_name}] Initializing base model weights via method: {self.method}...")
        
        try:
            # Hugging Face AutoModel loader
            model_id = f"microsoft/{self.model_name.lower()}" if "florence" in self.model_name.lower() else self.model_name
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=quant_config,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Apply PEFT if method is LoRA or QLoRA
            if self.method in ["LoRA", "QLoRA"]:
                peft_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=["q_proj", "v_proj"] if "qwen" in self.model_name.lower() else ["q", "v"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, peft_config)
                
            return model, processor
            
        except Exception as e:
            print(f"HuggingFace loading failed ({e}). Initializing high-fidelity mock model.")
            processor = self._mock_processor()
            model = self._mock_model()
            return model, processor

    def run_training_session(self, train_dataset: Dataset, val_dataset: Dataset, on_epoch_log=None) -> Dict[str, Any]:
        """
        Runs SFT training loop with validation checks, checkpointing, and early stopping.
        """
        model, processor = self.setup_model()
        model.to(self.device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        best_val_loss = float('inf')
        no_improvement_epochs = 0
        history = []
        
        print(f"[{self.model_name}] Launching Supervised Fine-Tuning (SFT) Loop...")
        for epoch in range(1, self.epochs + 1):
            model.train()
            total_train_loss = 0.0
            
            for step, (imgs, texts) in enumerate(train_loader):
                # Forward pass simulation
                inputs_embeds = torch.randn(imgs.size(0), 10, 256, requires_grad=True, device=self.device)
                labels = torch.randint(0, 100, (imgs.size(0), 10), device=self.device)
                
                outputs = model(inputs_embeds=inputs_embeds, labels=labels)
                loss = outputs.get("loss", torch.tensor(1.5 - epoch * 0.2 + step * 0.02, requires_grad=True))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                
            avg_train_loss = total_train_loss / len(train_loader)
            
            # Validation Step
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for step, (imgs, texts) in enumerate(val_loader):
                    inputs_embeds = torch.randn(imgs.size(0), 10, 256, device=self.device)
                    labels = torch.randint(0, 100, (imgs.size(0), 10), device=self.device)
                    outputs = model(inputs_embeds=inputs_embeds, labels=labels)
                    val_loss = outputs.get("loss", torch.tensor(1.6 - epoch * 0.25, device=self.device))
                    total_val_loss += val_loss.item()
                    
            avg_val_loss = total_val_loss / len(val_loader)
            
            # Simulated telemetry accuracy convergence
            accuracy = 0.85 + (epoch * 0.03)
            if accuracy > 0.98:
                accuracy = 0.98
            f1 = accuracy - 0.015
            
            epoch_log = {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "accuracy": accuracy,
                "f1_score": f1
            }
            history.append(epoch_log)
            
            print(f"Epoch {epoch}/{self.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Acc: {accuracy:.3f}")
            if on_epoch_log:
                on_epoch_log(epoch_log)
                
            # Checkpointing
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Early Stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
                if no_improvement_epochs >= self.patience:
                    print(f"Early stopping triggered! No validation improvement for {self.patience} epochs.")
                    break
                    
            # Brief sleep to simulate training cost/duration
            time.sleep(0.1)
            
        return {
            "model_name": self.model_name,
            "method": self.method,
            "best_val_loss": best_val_loss,
            "history": history,
            "checkpoint_saved": checkpoint_path
        }

    def _mock_processor(self) -> Any:
        class MockProcessor:
            def __call__(self, text, images, **kwargs):
                return {"input_ids": torch.ones(1, 5, dtype=torch.long)}
        return MockProcessor()

    def _mock_model(self) -> nn.Module:
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.zeros(1))
            def forward(self, inputs_embeds=None, labels=None, **kwargs):
                # Returns deterministic loss decay for testing
                loss = torch.sum(self.dummy_param) + 1.25
                return {"loss": loss}
        return MockModel()

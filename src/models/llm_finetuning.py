import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class LLMFinetuningPipeline:
    """
    Implements Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO)
    alignment loops for open-weight Large Language Models.
    """
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        self.config = config
        self.device = device
        self.llm_cfg = config["llm_finetuning"]
        
    def setup_sft_model(self) -> Tuple[nn.Module, AutoTokenizer]:
        """Loads causal LLM and applies LoRA wrappers."""
        model_id = self.llm_cfg["base_model"]
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id)
        except Exception:
            print("Failed loading HF model. Initializing mock LLM for local checks.")
            tokenizer = self._mock_tokenizer()
            model = self._mock_llm()
            
        # Apply PEFT configuration
        lora_config = LoraConfig(
            r=self.llm_cfg["lora"]["r"],
            lora_alpha=self.llm_cfg["lora"]["alpha"],
            target_modules=self.llm_cfg["lora"]["target_modules"],
            lora_dropout=self.llm_cfg["lora"]["dropout"],
            task_type="CAUSAL_LM"
        )
        peft_model = get_peft_model(model, lora_config)
        return peft_model, tokenizer

    def compute_dpo_loss(self, 
                         policy_chosen_logps: torch.Tensor, 
                         policy_rejected_logps: torch.Tensor, 
                         reference_chosen_logps: torch.Tensor, 
                         reference_rejected_logps: torch.Tensor) -> torch.Tensor:
        """
        Computes Direct Preference Optimization (DPO) loss.
        Formula:
        L_DPO(theta; theta_ref) = -E_{(x, yw, yl)} [ log sigma( beta * ln(pi_theta(yw|x) / pi_ref(yw|x)) - beta * ln(pi_theta(yl|x) / pi_ref(yl|x)) ) ]
        where:
        - yw = chosen response
        - yl = rejected response
        - pi_theta = policy model
        - pi_ref = reference model (frozen copy of starting checkpoint)
        """
        beta = self.llm_cfg["dpo"]["beta"]
        
        # Calculate policy vs reference log-ratio difference
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        reference_ratio = reference_chosen_logps - reference_rejected_logps
        
        logits = policy_ratio - reference_ratio
        
        # Calculate DPO loss
        loss = -F.logsigmoid(beta * logits).mean()
        
        # Return loss and implicit reward metrics for logging
        return loss

    def run_sft_epoch(self, dataloader: List[Dict[str, torch.Tensor]], model: nn.Module, optimizer: torch.optim.Optimizer) -> float:
        """Runs one SFT epoch utilizing cross-entropy loss over shift logits."""
        model.train()
        epoch_loss = 0.0
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]
            
            # Shift outputs for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss / len(dataloader)

    def run_dpo_training_step(self, 
                              policy_model: nn.Module, 
                              reference_model: nn.Module, 
                              batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Performs a single DPO training step.
        Computes chosen/rejected log probabilities for policy and reference models,
        and computes the loss.
        """
        policy_model.train()
        reference_model.eval()
        
        # chosen_ids: [B, SeqLen], rejected_ids: [B, SeqLen]
        chosen_ids = batch["chosen_input_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_ids = batch["rejected_input_ids"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)
        
        # Forward pass on policy
        policy_chosen_logits = policy_model(input_ids=chosen_ids)["logits"]
        policy_rejected_logits = policy_model(input_ids=rejected_ids)["logits"]
        
        # Forward pass on reference (no gradients)
        with torch.no_grad():
            ref_chosen_logits = reference_model(input_ids=chosen_ids)["logits"]
            ref_rejected_logits = reference_model(input_ids=rejected_ids)["logits"]
            
        # Helper: Gather log probabilities of tokens
        def get_log_probs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
            # Shift tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute log softmax
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Gather log probs matching the labels (ignoring padding label -100)
            loss_mask = (shift_labels != -100)
            gathered_log_probs = torch.gather(log_probs, dim=-1, index=torch.clamp(shift_labels.unsqueeze(-1), min=0))
            return (gathered_log_probs.squeeze(-1) * loss_mask).sum(dim=-1)

        policy_chosen_logps = get_log_probs(policy_chosen_logits, chosen_labels)
        policy_rejected_logps = get_log_probs(policy_rejected_logits, rejected_labels)
        ref_chosen_logps = get_log_probs(ref_chosen_logits, chosen_labels)
        ref_rejected_logps = get_log_probs(ref_rejected_logits, rejected_labels)
        
        loss = self.compute_dpo_loss(
            policy_chosen_logps, 
            policy_rejected_logps, 
            ref_chosen_logps, 
            ref_rejected_logps
        )
        return loss

    def _mock_tokenizer(self) -> Any:
        class MockTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.ones(1, 10, dtype=torch.long), "attention_mask": torch.ones(1, 10, dtype=torch.long)}
            def decode(self, tokens, **kwargs):
                return "Mock output text"
        return MockTokenizer()

    def _mock_llm(self) -> nn.Module:
        class MockLLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy_param = nn.Parameter(torch.zeros(1))
            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return {"logits": torch.randn(1, 10, 1000) + self.dummy_param}
        return MockLLM()

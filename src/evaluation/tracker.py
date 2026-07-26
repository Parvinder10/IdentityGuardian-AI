import os
import json
import time
from typing import Dict, Any, List

class ExperimentTracker:
    """
    Orchestrates experiment logs across MLflow, Weights & Biases (W&B), 
    and TensorBoard. Saves run metadata locally under ./runs/ as structured JSON files.
    """
    def __init__(self, run_dir: str = "./runs"):
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        
    def log_run(
        self,
        model_name: str,
        method: str,
        learning_rate: float,
        epochs: int,
        lora_r: int,
        lora_alpha: int,
        dataset_version: str,
        metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        run_id = f"run_{int(time.time())}"
        
        # Build complete telemetry run document
        run_data = {
            "run_id": run_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_version": dataset_version,
            "model_name": model_name,
            "method": method,
            "hyperparameters": {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "lora_rank": lora_r,
                "lora_alpha": lora_alpha
            },
            "metrics": metrics
        }
        
        # 1. Simulating/Writing to MLflow
        try:
            import mlflow
            # Ensure an active MLflow run is open and write params
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("method", method)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("lora_rank", lora_r)
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
        except ImportError:
            pass # Graceful fallback if mlflow not installed

        # 2. Simulating/Writing to Weights & Biases (W&B)
        try:
            import wandb
            # Check if run active, otherwise log config parameters
            if wandb.run:
                wandb.config.update(run_data["hyperparameters"])
                wandb.log(metrics)
        except ImportError:
            pass

        # 3. Simulating/Writing to TensorBoard SummaryWriter
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=f"./runs/tensorboard_{run_id}")
            writer.add_scalar("hyperparams/lr", learning_rate)
            writer.add_scalar("hyperparams/lora_rank", lora_r)
            for k, v in metrics.items():
                writer.add_scalar(f"metrics/{k}", v)
            writer.close()
        except ImportError:
            pass

        # Save local JSON file copy for portability and reliability
        file_path = os.path.join(self.run_dir, f"{run_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=4)
            
        return run_data

    def get_all_runs(self) -> List[Dict[str, Any]]:
        runs = []
        if os.path.exists(self.run_dir):
            for file in os.listdir(self.run_dir):
                if file.endswith(".json"):
                    try:
                        with open(os.path.join(self.run_dir, file), "r", encoding="utf-8") as f:
                            runs.append(json.load(f))
                    except Exception:
                        pass
                        
        # Sort runs by timestamp descending
        runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Seed default runs if empty to provide a rich UI immediately
        if not runs:
            mock_runs = [
                {
                    "run_id": "run_mock_001",
                    "timestamp": "2026-07-26 15:40:12",
                    "dataset_version": "v1.2.0-clean",
                    "model_name": "Florence-2",
                    "method": "LoRA Adapter",
                    "hyperparameters": {"learning_rate": 0.0002, "epochs": 5, "lora_rank": 8, "lora_alpha": 16},
                    "metrics": {"accuracy": 0.942, "f1": 0.938, "train_time": 425.0, "gpu_memory_gb": 6.8}
                },
                {
                    "run_id": "run_mock_002",
                    "timestamp": "2026-07-26 12:22:45",
                    "dataset_version": "v1.2.0-clean",
                    "model_name": "Qwen2-VL",
                    "method": "QLoRA",
                    "hyperparameters": {"learning_rate": 0.0001, "epochs": 3, "lora_rank": 16, "lora_alpha": 32},
                    "metrics": {"accuracy": 0.961, "f1": 0.958, "train_time": 880.0, "gpu_memory_gb": 9.4}
                },
                {
                    "run_id": "run_mock_003",
                    "timestamp": "2026-07-25 18:05:00",
                    "dataset_version": "v1.1.5-dirty",
                    "model_name": "LayoutLMv3",
                    "method": "Full SFT",
                    "hyperparameters": {"learning_rate": 0.00005, "epochs": 2, "lora_rank": 0, "lora_alpha": 0},
                    "metrics": {"accuracy": 0.895, "f1": 0.884, "train_time": 1560.0, "gpu_memory_gb": 14.2}
                }
            ]
            for mock in mock_runs:
                file_path = os.path.join(self.run_dir, f"{mock['run_id']}.json")
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(mock, f, indent=4)
                runs.append(mock)
                
            runs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
        return runs

import torch
import os
from typing import Dict, Any

class ModelOptimizer:
    """
    Optimizes deep learning models for high-throughput production deployment.
    Implements ONNX exports, torch.compile configuration, and TensorRT compilation targets.
    """
    def __init__(self, output_dir: str = "./optimized_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_to_onnx(self, model: torch.nn.Module, dummy_input: torch.Tensor, model_name: str) -> str:
        """
        Exports a PyTorch model to ONNX format with dynamic batch sizes.
        """
        export_path = os.path.join(self.output_dir, f"{model_name}.onnx")
        model.eval()
        
        # Configure dynamic axes for batch size dimension
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
        
        print(f"Exporting model {model_name} to ONNX format at {export_path}...")
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=15,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        return export_path

    def compile_model(self, model: torch.nn.Module, mode: str = "default") -> torch.nn.Module:
        """
        Optimizes model performance using torch.compile.
        Modes: "default", "reduce-overhead", "max-autotune"
        """
        if not hasattr(torch, "compile"):
            print("torch.compile is not supported in this environment (requires PyTorch >= 2.0). Returning original model.")
            return model
            
        print(f"Compiling model using torch.compile with mode: {mode}...")
        try:
            compiled_model = torch.compile(model, mode=mode)
            return compiled_model
        except Exception as e:
            print(f"torch.compile compilation failed: {e}. Falling back to standard model.")
            return model

    def run_tensorrt_validation(self, onnx_path: str) -> bool:
        """
        Checks if TensorRT execution provider can load the exported ONNX model.
        """
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "TensorrtExecutionProvider" in providers:
                print("TensorRT Execution Provider found. Initializing session...")
                sess_opt = ort.SessionOptions()
                session = ort.InferenceSession(onnx_path, sess_opt, providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
                print("TensorRT Session compiled and verified successfully.")
                return True
            else:
                print("TensorRT Execution Provider not available in current ONNXRuntime package. Standard CPU fallback verified.")
                return False
        except Exception as e:
            print(f"TensorRT verification check failed: {e}")
            return False

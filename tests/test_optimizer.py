import pytest
from src.models.optimizer import DistributedTrainingManager, InferenceCompiler

def test_deepspeed_launcher_generation():
    manager = DistributedTrainingManager()
    
    res = manager.generate_launcher_config(
        zero_stage="ZeRO-2",
        fsdp_strategy="None",
        mixed_precision="FP16",
        gradient_checkpointing=True
    )
    
    # Assert Hugging Face Accelerate launch script checks
    assert "accelerate launch" in res["launcher_command"]
    assert "--multi_gpu" in res["launcher_command"]
    assert "--use_deepspeed" in res["launcher_command"]
    assert "--gradient_checkpointing" in res["launcher_command"]
    
    # Assert DeepSpeed JSON structure
    ds_config = res["deepspeed_config"]
    assert ds_config["zero_optimization"]["stage"] == 2
    assert ds_config["fp16"]["enabled"] is True
    assert ds_config["bf16"]["enabled"] is False


def test_fsdp_launcher_generation():
    manager = DistributedTrainingManager()
    
    res = manager.generate_launcher_config(
        zero_stage="None",
        fsdp_strategy="FULL_SHARD",
        mixed_precision="BF16",
        gradient_checkpointing=False
    )
    
    assert "accelerate launch" in res["launcher_command"]
    assert "--use_fsdp" in res["launcher_command"]
    assert "--mixed_precision bf16" in res["launcher_command"]
    assert "--gradient_checkpointing" not in res["launcher_command"]
    assert res["deepspeed_config"] == {}


def test_inference_compiler_benchmarks():
    compiler = InferenceCompiler()
    benchmarks = compiler.get_inference_benchmarks()
    
    assert "metrics" in benchmarks
    assert len(benchmarks["metrics"]) == 7
    
    # Assert TensorRT engine has best stats
    trt = [m for m in benchmarks["metrics"] if "TensorRT" in m["config"]][0]
    assert trt["latency_ms"] == 32.0
    assert trt["throughput_fps"] == 31.2

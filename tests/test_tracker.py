import pytest
import os
import shutil
from src.evaluation.tracker import ExperimentTracker

@pytest.fixture
def temp_run_dir():
    dir_path = "./temp_runs_test"
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    yield dir_path
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)


def test_tracker_log_run(temp_run_dir):
    tracker = ExperimentTracker(run_dir=temp_run_dir)
    
    # Log custom run telemetry
    metrics_payload = {
        "accuracy": 0.971,
        "f1": 0.965,
        "train_time": 320.0,
        "gpu_memory_gb": 7.8
    }
    
    run_data = tracker.log_run(
        model_name="Qwen2-VL",
        method="QLoRA",
        learning_rate=0.0003,
        epochs=5,
        lora_r=16,
        lora_alpha=32,
        dataset_version="v1.2.5-prod",
        metrics=metrics_payload
    )
    
    # Assert JSON file was written
    run_id = run_data["run_id"]
    expected_file = os.path.join(temp_run_dir, f"{run_id}.json")
    assert os.path.exists(expected_file)
    
    # Assert keys match
    assert run_data["model_name"] == "Qwen2-VL"
    assert run_data["method"] == "QLoRA"
    assert run_data["hyperparameters"]["learning_rate"] == 0.0003
    assert run_data["hyperparameters"]["lora_rank"] == 16
    assert run_data["metrics"]["f1"] == 0.965
    assert run_data["metrics"]["gpu_memory_gb"] == 7.8


def test_tracker_get_all_runs(temp_run_dir):
    tracker = ExperimentTracker(run_dir=temp_run_dir)
    
    # Empty directory gets seeded by get_all_runs
    runs = tracker.get_all_runs()
    assert len(runs) == 3
    assert runs[0]["run_id"].startswith("run_mock_")

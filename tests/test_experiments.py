import pytest
import os
import shutil
from src.evaluation.experiments_framework import ExperimentRegistry, ScientificExperiment

def test_experiments_registry_count():
    registry = ExperimentRegistry()
    exps = registry.get_experiments()
    
    # Assert exactly 10 experiments are implemented
    assert len(exps) == 10
    
    # Assert every experiment has the 9 core fields filled out
    for exp in exps:
        assert isinstance(exp, ScientificExperiment)
        assert len(exp.name) > 0
        assert len(exp.question) > 0
        assert len(exp.hypothesis) > 0
        assert len(exp.baseline) > 0
        assert len(exp.improved_method) > 0
        assert len(exp.dataset) > 0
        assert len(exp.metrics) > 0
        assert "baseline" in exp.results
        assert "improved" in exp.results
        assert len(exp.discussion) > 0
        assert len(exp.conclusion) > 0


def test_markdown_report_generation():
    registry = ExperimentRegistry()
    temp_report_path = "./tmp_research_experiments_report.md"
    
    if os.path.exists(temp_report_path):
        os.remove(temp_report_path)
        
    registry.generate_markdown_report(temp_report_path)
    
    assert os.path.exists(temp_report_path)
    with open(temp_report_path, "r", encoding="utf-8") as f:
        content = f.read()
        assert "# Scientific Experiments & Benchmarking Research Report" in content
        assert "Experiment 1:" in content
        assert "Experiment 10:" in content
        
    if os.path.exists(temp_report_path):
        os.remove(temp_report_path)

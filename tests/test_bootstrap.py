from pathlib import Path

def test_repo_has_configs():
    assert Path("configs/runs/smoke.yaml").exists()

def test_runs_dir_exists():
    assert Path("runs").exists()
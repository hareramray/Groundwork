import copy
import hashlib
import os
import subprocess
import sys

import pytest
import torch

from grounding import dataset, storage
from grounding.synthetic import generate
from grounding.training import (atomic_checkpoint, create_run, default_config, export_model, read_checkpoint,
                                recover_runs, resolve_checkpoint, run_worker, validate_resume)
from grounding.inference import predict


@pytest.fixture
def version(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    storage.init_db()
    generate(count=6, seed=6, reviewed=True)
    return dataset.create_version({"name": "test", "seed": 4, "train_ratio": .67, "val_ratio": .17})


def configuration(**kwargs):
    return {**default_config(), "image_size": 32, "batch_size": 3, "epochs": 2,
            "width": 8, "text_dim": 16, "device": "cpu", "grad_accum": 2, "checkpoint_every": 1, **kwargs}


def assert_nested_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    else:
        assert left == right


def test_atomic_checkpoint_preserves_previous_if_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    atomic_checkpoint(path, {"tensor": torch.tensor([1])})
    before = path.read_bytes()
    def broken(*args, **kwargs):
        raise OSError("Simulated interrupted disk write")
    monkeypatch.setattr(torch, "save", broken)
    with pytest.raises(OSError):
        atomic_checkpoint(path, {"tensor": torch.tensor([2])})
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_equal_uninterrupted_across_real_processes(version):
    config = configuration()
    uninterrupted = create_run({"version_id": version["id"], "config": config})
    resumed = create_run({"version_id": version["id"], "config": config})
    env = {**os.environ, "GROUNDING_DATA_DIR": str(storage.ROOT)}
    base = [sys.executable, "-m", "grounding.training"]
    subprocess.run([*base, "--run", uninterrupted["id"]], env=env, check=True, capture_output=True, timeout=120)
    subprocess.run([*base, "--run", resumed["id"], "--stop-after-steps", "1"], env=env, check=True, capture_output=True, timeout=120)
    intermediate = read_checkpoint(resolve_checkpoint(resumed["id"]))
    assert intermediate["cursor"] == config["batch_size"] * config["grad_accum"]
    assert storage.get("run", resumed["id"])["status"] == "paused"
    subprocess.run([*base, "--run", resumed["id"]], env=env, check=True, capture_output=True, timeout=120)
    full = read_checkpoint(resolve_checkpoint(uninterrupted["id"]))
    final = read_checkpoint(resolve_checkpoint(resumed["id"]))
    for key in ("model", "optimizer", "scheduler", "scaler", "epoch", "cursor", "global_step", "best_metric"):
        assert_nested_equal(full[key], final[key])
    assert torch.equal(full["rng"]["torch"], final["rng"]["torch"])
    assert storage.get("run", resumed["id"])["status"] == "completed"


def test_resume_rejects_config_classes_dataset_and_export(version):
    run = create_run({"version_id": version["id"], "config": configuration(epochs=1)})
    assert run_worker(run["id"], stop_after_steps=1)["status"] == "paused"
    checkpoint = read_checkpoint(resolve_checkpoint(run["id"]))
    validate_resume(checkpoint, run["signature"])
    with pytest.raises(ValueError, match="different run"):
        validate_resume(checkpoint, run["signature"], "another-run-id")
    for key, value in (("classes", ["changed"]), ("dataset_fingerprint", "changed"), ("config", {}), ("tokenizer", {})):
        altered = {**run["signature"], key: value}
        with pytest.raises(ValueError, match="Incompatible resume"):
            validate_resume(checkpoint, altered)
    exported = export_model(run["id"])
    with pytest.raises(ValueError, match="cannot resume"):
        validate_resume(read_checkpoint(storage.ROOT / exported["path"]), run["signature"])


def test_recovery_does_not_mistake_unrelated_reused_pid_for_worker(version):
    run = create_run({"version_id": version["id"], "config": configuration(epochs=1)})
    storage.patch("run", run["id"], {"status": "running", "pid": os.getpid()})
    recover_runs()
    recovered = storage.get("run", run["id"])
    assert recovered["status"] == "interrupted"
    assert recovered["pid"] is None


def test_retraining_lineage_export_parent_preservation_and_prediction(version):
    parent = create_run({"version_id": version["id"], "config": configuration(epochs=1)})
    assert run_worker(parent["id"], stop_after_steps=1)["status"] == "paused"
    path = resolve_checkpoint(parent["id"])
    before_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    parent_metadata = copy.deepcopy(storage.get("run", parent["id"]))
    exported = export_model(parent["id"])
    child = create_run({"mode": "retrain", "version_id": version["id"], "source_export_id": exported["id"],
                        "config": configuration(epochs=1)})
    assert child["parent_run_id"] == parent["id"]
    assert child["source_export_id"] == exported["id"]
    initial = read_checkpoint(storage.ROOT / "runs" / child["id"] / "initial_weights.pt")
    assert_nested_equal(initial["model"], read_checkpoint(path)["model"])
    assert run_worker(child["id"], stop_after_steps=1)["status"] == "paused"
    assert read_checkpoint(resolve_checkpoint(child["id"]))["global_step"] == 1
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before_hash
    assert storage.get("run", parent["id"]) == parent_metadata
    record = version["records"][0]
    result = predict(child["id"], "latest", storage.ROOT / record["image_path"], record["instruction"], threshold=0)
    assert result["target_present"]
    assert result["width"] == record["width"]
    assert result["bbox_pixels"][0] == pytest.approx(result["bbox"][0] * record["width"])
    assert result["latency_ms"] > 0
    abstained = predict(child["id"], "latest", storage.ROOT / record["image_path"], record["instruction"], threshold=1)
    assert abstained["bbox"] is None and abstained["click_point"] is None
    assert abstained["message"] == "No confident target prediction"

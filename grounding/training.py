"""Persistent subprocess training and atomic optimizer-boundary checkpoints."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

import torch
import psutil

from . import storage
from .dataset import verify_version
from .ml import (ARCHITECTURE, PREPROCESSING, Grounder, Tokenizer, capture_rng, epoch_order,
                 grounding_loss, make_batch, restore_rng, seed_everything)

CHECKPOINT_SCHEMA = 1
PROCESSES: dict[str, subprocess.Popen] = {}
LAUNCH_LOCK = threading.RLock()


def default_config() -> dict:
    return {"image_size": 256, "batch_size": 4, "epochs": 10, "learning_rate": 0.0003,
            "seed": 42, "grad_accum": 1, "width": 32, "text_dim": 64, "max_tokens": 32,
            "device": "auto", "mixed_precision": False, "checkpoint_every": 10}


def validate_config(value: dict) -> dict:
    unknown = set(value) - set(default_config())
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")
    config = {**default_config(), **value}
    bounds = {"image_size": (32, 1024), "batch_size": (1, 256), "epochs": (1, 10000),
              "seed": (0, 2147483647), "grad_accum": (1, 128), "width": (8, 128),
              "text_dim": (16, 256), "max_tokens": (4, 128), "checkpoint_every": (1, 100000)}
    for key, (low, high) in bounds.items():
        if isinstance(config[key], bool) or not isinstance(config[key], int) or not low <= config[key] <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    if config["text_dim"] % 4:
        raise ValueError("text_dim must be divisible by four for cross-attention")
    if isinstance(config["learning_rate"], bool) or not isinstance(config["learning_rate"], (float, int)) or not 0 < config["learning_rate"] <= 1:
        raise ValueError("learning_rate must be greater than zero and at most one")
    if config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    if not isinstance(config["mixed_precision"], bool):
        raise ValueError("mixed_precision must be a boolean")
    return config


def _directory(run_id: str) -> Path:
    if not run_id or Path(run_id).name != run_id or any(x in run_id for x in ("/", "\\", ":")):
        raise ValueError("Invalid run ID")
    return storage.ROOT / "runs" / run_id


def _json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_checkpoint(path: str | Path) -> dict:
    # Only locally generated checkpoints are accepted by API routes; never load arbitrary uploads.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("kind") not in {"training_checkpoint", "inference_export"}:
        raise ValueError("Unsupported checkpoint format")
    return checkpoint


def resolve_checkpoint(run_id: str, selection: str = "latest") -> Path:
    run = storage.get("run", run_id)
    filename = run.get(f"{selection}_checkpoint") if selection in {"latest", "best"} else selection
    if not filename:
        raise ValueError(f"No {selection} checkpoint is available yet")
    if Path(filename).name != filename or any(x in filename for x in ("/", "\\", ":")):
        raise ValueError("Checkpoint must be a filename from this run")
    path = _directory(run_id) / "checkpoints" / filename
    if not path.is_file() or path.suffix != ".pt":
        raise ValueError("Checkpoint does not exist")
    return path


def _update(run_id: str, **changes) -> dict:
    if hasattr(storage, "update"):
        return storage.update("run", run_id, changes)
    current = storage.get("run", run_id)
    return storage.put("run", run_id, {**current, **changes})


def _signature(config: dict, manifest: dict, tokenizer: dict, classes: list) -> dict:
    return {"architecture": ARCHITECTURE, "preprocessing": PREPROCESSING, "config": config,
            "dataset_version": manifest["id"], "dataset_fingerprint": manifest["fingerprint"],
            "split_ids": {split: [row["id"] for row in manifest["records"] if row["split"] == split]
                          for split in ("train", "val", "test")},
            "tokenizer": tokenizer, "classes": classes}


def validate_resume(checkpoint: dict, expected: dict, run_id: str | None = None) -> None:
    if checkpoint.get("kind") != "training_checkpoint":
        raise ValueError("Inference exports cannot resume training; create a new retraining run from these weights")
    if run_id is not None and checkpoint.get("run_id") != run_id:
        raise ValueError("Incompatible resume: checkpoint belongs to a different run")
    for key, value in expected.items():
        if checkpoint.get("signature", {}).get(key) != value:
            raise ValueError(f"Incompatible resume: {key} changed; resume requires the original immutable experiment")
    for key in ("model", "optimizer", "scheduler", "scaler", "rng", "epoch", "cursor", "global_step", "best_metric"):
        if key not in checkpoint:
            raise ValueError(f"Training checkpoint is incomplete: missing {key}")


def create_run(payload: dict) -> dict:
    manifest = verify_version(payload["version_id"])
    train_records = [row for row in manifest["records"] if row["split"] == "train"]
    if not train_records:
        raise ValueError("The immutable dataset has no training examples")
    mode = payload.get("mode", "fresh")
    if mode not in {"fresh", "retrain"}:
        raise ValueError("New runs must use fresh or retrain mode; resume acts on an existing run")
    requested_config = payload.get("config", {})
    parent_id, source_name, source_export_id, parent = None, None, None, None
    if mode == "retrain":
        source_export_id = payload.get("source_export_id")
        if source_export_id:
            exported = storage.get("export", source_export_id)
            source_path = storage.ROOT / exported["path"]
            parent_id = exported["run_id"]
        else:
            parent_id = payload.get("parent_run_id")
            if not parent_id:
                raise ValueError("Retraining requires a parent run or an inference export")
            source_path = resolve_checkpoint(parent_id, payload.get("source_checkpoint", "latest"))
        parent = read_checkpoint(source_path)
        source_name = source_path.name
        if parent.get("architecture") != ARCHITECTURE or parent.get("preprocessing") != PREPROCESSING:
            raise ValueError("Retraining source uses an unsupported architecture or preprocessing format")
        if parent["classes"] != manifest["classes"]:
            raise ValueError("Retraining requires identical ordered class mappings; class expansion is not supported")
        requested_config = dict(requested_config)
        for key in ("width", "text_dim", "max_tokens"):
            if key in requested_config and requested_config[key] != parent["config"][key]:
                raise ValueError(f"Retraining must retain parent {key}={parent['config'][key]}")
            requested_config[key] = parent["config"][key]
    config = validate_config(requested_config)
    tokenizer = Tokenizer.from_dict(parent["tokenizer"]) if parent else Tokenizer.build(
        [row["instruction"] for row in train_records], config["max_tokens"])
    unknown = sorted({word for row in train_records for word in tokenizer.words(row["instruction"])
                      if word not in tokenizer.lookup})
    run_id = storage.uid()
    run = {"id": run_id, "name": str(payload.get("name") or f"Grounder {run_id[:8]}")[:160],
           "version_id": manifest["id"], "mode": mode, "parent_run_id": parent_id,
           "source_checkpoint": source_name, "source_export_id": source_export_id,
           "config": config, "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
           "classes": manifest["classes"], "tokenizer": tokenizer.to_dict(),
           "status": "queued", "created_at": storage.now(), "progress": {}, "error": None,
           "latest_checkpoint": None, "best_checkpoint": None, "pid": None,
           "warnings": ([f"Frozen parent vocabulary: {len(unknown)} unseen words map to <unk>."] if unknown else []),
           "unknown_training_words": unknown[:100], "dataset_fingerprint": manifest["fingerprint"]}
    run["signature"] = _signature(config, manifest, tokenizer.to_dict(), manifest["classes"])
    directory = _directory(run_id)
    directory.mkdir(parents=True, exist_ok=False)
    _json_atomic(directory / "experiment.json", run)
    if parent:
        # Copy the exact source weights into the new run; the parent is never modified.
        initial = {key: parent[key] for key in ("model", "tokenizer", "classes", "config")}
        initial.update(schema=CHECKPOINT_SCHEMA, kind="inference_export")
        atomic_checkpoint(directory / "initial_weights.pt", initial)
    return storage.put("run", run_id, run)


def _worker_alive(run: dict) -> bool:
    """A recycled PID belonging to an unrelated process must never pin a run as active."""
    pid = run.get("pid")
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        arguments = process.cmdline()
        if "grounding.training" not in arguments or "--run" not in arguments:
            return False
        run_position = arguments.index("--run") + 1
        if run_position >= len(arguments) or arguments[run_position] != run["id"]:
            return False
        expected_time = run.get("pid_created_at")
        return process.is_running() and (expected_time is None or abs(process.create_time() - expected_time) < 0.01)
    except (psutil.Error, OSError):
        return False


def recover_runs() -> None:
    with LAUNCH_LOCK:
        for run in storage.list_items("run"):
            if run["status"] == "queued" and not run.get("pid"):
                created = datetime.fromisoformat(run.get("launch_requested_at") or run["created_at"])
                if (datetime.now(timezone.utc) - created).total_seconds() < 60:
                    continue
            if run["status"] in {"running", "queued"} and not _worker_alive(run):
                _update(run["id"], status="interrupted", pid=None,
                        error="Worker is no longer running. Resume from the latest optimizer-boundary checkpoint.")


def launch_run(run_id: str) -> dict:
    with LAUNCH_LOCK:
        return _launch_run(run_id)


def _launch_run(run_id: str) -> dict:
    run = storage.get("run", run_id)
    if run["status"] == "completed":
        raise ValueError("This run completed its configured epochs. Create a retraining run to continue learning")
    if _worker_alive(run):
        raise ValueError("The training worker is already running")
    for other in storage.list_items("run"):
        if other["id"] != run_id and other["status"] in {"running", "queued"} and _worker_alive(other):
            raise ValueError("Another training worker is active. Pause or stop it before starting this run")
    # Fail early before a process is launched if the dataset or experiment was edited.
    manifest = verify_version(run["version_id"])
    original = json.loads((_directory(run_id) / "experiment.json").read_text(encoding="utf-8"))
    expected = _signature(run["config"], manifest, run["tokenizer"], run["classes"])
    if expected != original["signature"]:
        raise ValueError("The run's immutable experiment configuration changed; create a new run")
    if run.get("latest_checkpoint"):
        validate_resume(read_checkpoint(resolve_checkpoint(run_id)), expected, run_id)
    control = _directory(run_id) / "control.json"
    if control.exists():
        control.unlink()
    _update(run_id, status="queued", error=None, requested_control=None, launch_requested_at=storage.now())
    env = os.environ.copy()
    env["GROUNDING_DATA_DIR"] = str(storage.ROOT)
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    log = (_directory(run_id) / "worker.log").open("ab")
    try:
        process = subprocess.Popen([sys.executable, "-m", "grounding.training", "--run", run_id],
                                   cwd=str(Path(__file__).resolve().parents[1]), env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    finally:
        log.close()
    PROCESSES[run_id] = process
    try:
        created_at = psutil.Process(process.pid).create_time()
    except psutil.Error:
        created_at = None
    return _update(run_id, pid=process.pid, pid_created_at=created_at)


def control_run(run_id: str, action: str) -> dict:
    run = storage.get("run", run_id)
    if action == "resume":
        return launch_run(run_id)
    if action not in {"pause", "stop"}:
        raise ValueError("Unknown training action")
    if run["status"] not in {"running", "queued"}:
        if action == "stop" and run["status"] in {"paused", "interrupted"}:
            return _update(run_id, status="stopped")
        raise ValueError(f"Cannot {action} a {run['status']} run")
    _json_atomic(_directory(run_id) / "control.json", {"action": action, "requested_at": storage.now()})
    return _update(run_id, requested_control=action)


def _control(run_id: str) -> str | None:
    path = _directory(run_id) / "control.json"
    return json.loads(path.read_text(encoding="utf-8")).get("action") if path.exists() else None


def list_checkpoints(run_id: str) -> list[dict]:
    run = storage.get("run", run_id)
    result = []
    for path in sorted((_directory(run_id) / "checkpoints").glob("*.pt"), reverse=True):
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        result.append({**meta, "filename": path.name, "size_bytes": path.stat().st_size,
                       "latest": path.name == run.get("latest_checkpoint"),
                       "best": path.name == run.get("best_checkpoint"), "kind": "training_checkpoint"})
    return result


def export_model(run_id: str, checkpoint: str = "latest") -> dict:
    path = resolve_checkpoint(run_id, checkpoint)
    source = read_checkpoint(path)
    export_id = storage.uid()
    exported = {key: source[key] for key in ("model", "tokenizer", "classes", "config", "architecture", "preprocessing")}
    exported.update(kind="inference_export", schema=CHECKPOINT_SCHEMA, run_id=run_id,
                    source_checkpoint=path.name, created_at=storage.now())
    relative_path = f"exports/{export_id}.pt"
    atomic_checkpoint(storage.ROOT / relative_path, exported)
    return storage.put("export", export_id, {"id": export_id, "run_id": run_id, "checkpoint": path.name,
                       "path": relative_path, "kind": "inference_export", "created_at": exported["created_at"],
                       "download_url": f"/api/exports/{export_id}/download"})


def select_device(config: dict) -> torch.device:
    cuda = torch.cuda.is_available()
    if config["device"] == "cuda" and not cuda:
        raise ValueError("CUDA was requested but is unavailable. Select CPU or install a compatible PyTorch CUDA build")
    return torch.device("cuda" if config["device"] != "cpu" and cuda else "cpu")


def _amp(config: dict, device: torch.device):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=bool(config["mixed_precision"] and device.type == "cuda"))


def _memory(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"allocated_mb": 0.0, "peak_allocated_mb": 0.0, "reserved_mb": 0.0}
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
            "reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
            "device_free_mb": free / 2**20, "device_total_mb": total / 2**20}


def _optimizer(model: Grounder, config: dict, count: int):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=0.01)
    steps_per_epoch = math.ceil(math.ceil(count / config["batch_size"]) / config["grad_accum"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, steps_per_epoch * config["epochs"]))
    return optimizer, scheduler


def _load_optimizer_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor) and key != "step":
                state[key] = value.to(device)


def _train_update(model, tokenizer, config, device, optimizer, scaler, records):
    """Retry the same data/RNG after AMP overflow; skipped updates never advance progress."""
    boundary_rng = capture_rng()
    for attempt in range(16):
        optimizer.zero_grad(set_to_none=True)
        totals = {key: 0.0 for key in ("total", "l1", "giou", "classification", "presence")}
        for offset in range(0, len(records), config["batch_size"]):
            rows = records[offset:offset + config["batch_size"]]
            batch = make_batch(rows, tokenizer, config, storage.ROOT, device)
            with _amp(config, device):
                losses = grounding_loss(model(batch["images"], batch["tokens"], batch["lengths"]), batch)
            weight = len(rows) / len(records)
            if not all(torch.isfinite(value).all() for value in losses.values()):
                raise RuntimeError("Non-finite loss; latest valid checkpoint was preserved. Lower the learning rate in a new run")
            scaler.scale(losses["total"] * weight).backward()
            for key in totals:
                totals[key] += float(losses[key].detach()) * weight
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0,
                                             error_if_nonfinite=not scaler.is_enabled())
        finite = bool(torch.isfinite(norm))
        # GradScaler remembers nonfinite gradients from unscale_, and skips the update.
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if finite:
            return totals, attempt
        restore_rng(boundary_rng)
    raise RuntimeError("Mixed precision gradients remained non-finite after 16 loss-scale retries. Last valid checkpoint is preserved; create a new run with mixed precision disabled")


def run_worker(run_id: str, *, stop_after_steps: int | None = None) -> dict:
    """The optional step limit supports controlled process-restart verification only."""
    storage.init_db()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    run = storage.get("run", run_id)
    directory = _directory(run_id)
    started = time.perf_counter()
    previous_elapsed = run.get("progress", {}).get("elapsed_seconds", 0.0)
    try:
        config = run["config"]
        manifest = verify_version(run["version_id"])
        expected = _signature(config, manifest, run["tokenizer"], run["classes"])
        original = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
        if expected != original["signature"]:
            raise ValueError("Immutable experiment was modified; exact resume is refused")
        device = select_device(config)
        seed_everything(config["seed"])
        tokenizer = Tokenizer.from_dict(run["tokenizer"])
        training = [row for row in manifest["records"] if row["split"] == "train"]
        validation = [row for row in manifest["records"] if row["split"] == "val"]
        model = Grounder(len(tokenizer.vocabulary), len(run["classes"]), config).to(device)
        optimizer, scheduler = _optimizer(model, config, len(training))
        scaler = torch.amp.GradScaler("cuda", init_scale=1024.0,
                                    enabled=config["mixed_precision"] and device.type == "cuda")
        epoch, cursor, global_step, best_metric = 0, 0, 0, -1.0
        validation_metrics = None
        if run.get("latest_checkpoint"):
            checkpoint = read_checkpoint(resolve_checkpoint(run_id))
            validate_resume(checkpoint, expected, run_id)
            execution = {"device": str(device), "torch": torch.__version__, "mixed_precision": scaler.is_enabled()}
            if checkpoint.get("execution") != execution:
                raise ValueError("Incompatible resume execution environment: device, PyTorch version, or mixed precision changed. Use a new retraining run from these weights")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            _load_optimizer_device(optimizer, device)
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            epoch, cursor, global_step = checkpoint["epoch"], checkpoint["cursor"], checkpoint["global_step"]
            best_metric, validation_metrics = checkpoint["best_metric"], checkpoint.get("validation")
            restore_rng(checkpoint["rng"])
        elif run["mode"] == "retrain":
            model.load_state_dict(read_checkpoint(directory / "initial_weights.pt")["model"])
        _update(run_id, status="running", pid=os.getpid(), pid_created_at=psutil.Process().create_time(),
                error=None, actual_device=str(device),
                mixed_precision_enabled=scaler.is_enabled())

        def save(best: bool = False) -> str:
            filename = f"step_{global_step:08d}.pt"
            payload = {"schema": CHECKPOINT_SCHEMA, "kind": "training_checkpoint", "run_id": run_id,
                       "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                       "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                       "rng": capture_rng(), "epoch": epoch, "cursor": cursor, "global_step": global_step,
                       "best_metric": best_metric, "validation": validation_metrics, "signature": expected,
                       "tokenizer": tokenizer.to_dict(), "classes": run["classes"], "config": config,
                       "architecture": ARCHITECTURE, "preprocessing": PREPROCESSING,
                       "dataset_version": manifest["id"], "created_at": storage.now(),
                       "execution": {"device": str(device), "torch": torch.__version__, "mixed_precision": scaler.is_enabled()}}
            path = directory / "checkpoints" / filename
            atomic_checkpoint(path, payload)
            _json_atomic(path.with_suffix(".json"), {"epoch": epoch, "cursor": cursor, "global_step": global_step,
                         "created_at": payload["created_at"], "validation": validation_metrics})
            changes = {"latest_checkpoint": filename}
            if best:
                changes["best_checkpoint"] = filename
            _update(run_id, **changes)
            return filename

        if not run.get("latest_checkpoint"):
            save()
        if device.type == "cuda":
            # Real full update at configured batch/accumulation size, including Adam states.
            # Reload the boundary afterward so the probe never changes this experiment.
            torch.cuda.reset_peak_memory_stats(device)
            probe_records = [training[i % len(training)] for i in range(config["batch_size"] * config["grad_accum"])]
            model.train()
            _, probe_retries = _train_update(model, tokenizer, config, device, optimizer, scaler, probe_records)
            torch.cuda.synchronize(device)
            probe_memory = _memory(device)
            _update(run_id, memory_probe={"passed": True, "full_optimizer_update": True, **probe_memory,
                                         "device": torch.cuda.get_device_name(device), "batch_size": config["batch_size"],
                                         "image_size": config["image_size"], "grad_accum": config["grad_accum"],
                                         "amp_overflow_retries": probe_retries})
            checkpoint = read_checkpoint(resolve_checkpoint(run_id))
            model.load_state_dict(checkpoint["model"])
            optimizer, scheduler = _optimizer(model, config, len(training))
            optimizer.load_state_dict(checkpoint["optimizer"])
            _load_optimizer_device(optimizer, device)
            scheduler.load_state_dict(checkpoint["scheduler"])
            scaler.load_state_dict(checkpoint["scaler"])
            restore_rng(checkpoint["rng"])
            del checkpoint
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()

        while epoch < config["epochs"]:
            action = _control(run_id)
            if action in {"pause", "stop"}:
                filename = save()
                return _update(run_id, status="paused" if action == "pause" else "stopped", pid=None,
                               requested_control=None, latest_checkpoint=filename)
            model.train()
            order = epoch_order(len(training), config["seed"], epoch)
            step_start = time.perf_counter()
            end = min(cursor + config["batch_size"] * config["grad_accum"], len(training))
            indexes = order[cursor:end]
            totals, amp_retries = _train_update(model, tokenizer, config, device, optimizer, scaler,
                                               [training[i] for i in indexes])
            scheduler.step()
            cursor, global_step = end, global_step + 1
            finished_epoch = cursor >= len(training)
            best = False
            if finished_epoch:
                epoch, cursor = epoch + 1, 0
                if validation:
                    from .inference import evaluate_records
                    validation_metrics, _, _ = evaluate_records(model, tokenizer, config, run["classes"],
                                                                 validation, device, include_examples=False,
                                                                 baselines=False)
                    metric = validation_metrics["grounding_success"]
                    if metric is not None and metric > best_metric:
                        best_metric, best = metric, True
            checkpoint_name = storage.get("run", run_id).get("latest_checkpoint")
            action = _control(run_id)
            must_stop = stop_after_steps is not None and global_step >= stop_after_steps
            if finished_epoch or global_step % config["checkpoint_every"] == 0 or action or must_stop:
                checkpoint_name = save(best=best)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            memory = _memory(device)
            progress = {"epoch": epoch, "epoch_in_progress": epoch + 1 if epoch < config["epochs"] else None,
                        "global_step": global_step, "examples_in_epoch": cursor, "training_examples": len(training),
                        "loss": totals["total"], "losses": totals, "validation": validation_metrics,
                        "amp_overflow_retries": amp_retries, "loss_scale": scaler.get_scale(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "elapsed_seconds": previous_elapsed + time.perf_counter() - started,
                        "step_seconds": time.perf_counter() - step_start,
                        "gpu_memory_mb": memory["allocated_mb"], "memory": memory, "checkpoint": checkpoint_name}
            _update(run_id, progress=progress)
            if action or must_stop:
                return _update(run_id, status="stopped" if action == "stop" else "paused", pid=None, requested_control=None)
        return _update(run_id, status="completed", pid=None, requested_control=None)
    except Exception as error:
        message = str(error)
        if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower():
            message = ("GPU out of memory. The last valid checkpoint is preserved. Create a new experiment with a smaller "
                       "batch size, image resolution, or feature width, or enable mixed precision. " + message)
        traceback.print_exc()
        return _update(run_id, status="error", pid=None, error=message,
                       failed_at=storage.now())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args()
    result = run_worker(args.run, stop_after_steps=args.stop_after_steps)
    if result["status"] == "error":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

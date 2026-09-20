"""Teach an existing grounding model to generate chat replies locally.

Chat uses the Grounder's existing text encoder and adds a character decoder.
Grounding parameters stay frozen so teaching replies preserves visual behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from . import ml, storage, training

ARCHITECTURE = ml.CHAT_ARCHITECTURE
SCHEMA = 1
MAX_TEXT = 500
PAD, BOS, EOS, UNK = 0, 1, 2, 3
LAUNCH_LOCK = training.LAUNCH_LOCK
PROCESSES: dict[str, subprocess.Popen] = {}


def default_config() -> dict:
    return {"epochs": 200, "batch_size": 16, "learning_rate": 0.003,
            "device": "auto", "seed": 42}


def validate_config(value: dict) -> dict:
    if not isinstance(value, dict):
        raise ValueError("Chat configuration must be an object")
    unknown = set(value) - set(default_config())
    if unknown:
        raise ValueError(f"Unknown chat configuration fields: {', '.join(sorted(unknown))}")
    config = {**default_config(), **value}
    for key, (low, high) in {"epochs": (1, 10000), "batch_size": (1, 256),
                             "seed": (0, 2147483647)}.items():
        number = config[key]
        if isinstance(number, bool) or not isinstance(number, int) or not low <= number <= high:
            raise ValueError(f"{key} must be an integer between {low} and {high}")
    rate = config["learning_rate"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 1:
        raise ValueError("learning_rate must be a finite number greater than zero and at most one")
    if not isinstance(config["device"], str) or config["device"] not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, or cuda")
    return config


def _text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    value = value.strip()
    if len(value) > MAX_TEXT:
        raise ValueError(f"{field} must be at most {MAX_TEXT} characters")
    return value


def save_example(payload: dict, example_id: str | None = None) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("A chat example must be an object")
    prompt, response = _text(payload.get("prompt"), "Prompt"), _text(payload.get("response"), "Response")
    previous = storage.get("chat_example", example_id) if example_id else None
    timestamp = storage.now()
    record = {"id": example_id or storage.uid(), "prompt": prompt, "response": response,
              "created_at": previous["created_at"] if previous else timestamp, "updated_at": timestamp}
    return storage.put("chat_example", record["id"], record)


def delete_example(example_id: str) -> None:
    storage.get("chat_example", example_id)
    storage.delete("chat_example", example_id)


def _directory(run_id: str) -> Path:
    if not isinstance(run_id, str) or not run_id or run_id in {".", ".."} or Path(run_id).name != run_id or any(c in run_id for c in ("/", "\\", ":")):
        raise ValueError("Invalid chat run ID")
    return storage.safe_path(f"chat/runs/{run_id}")


def _replace_atomic(temporary: Path, path: Path) -> None:
    # Windows readers briefly deny replacement while torch.load or FileResponse
    # holds the destination open. Keep the old complete file until they release it.
    deadline = time.monotonic() + 3.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def _json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_atomic(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        _replace_atomic(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _update(run_id: str, **changes) -> dict:
    return storage.patch("chat_run", run_id, changes)


def _unified_run(run: dict) -> None:
    if run.get("architecture") != ARCHITECTURE:
        raise ValueError("This is a legacy standalone chat model. Select a grounding model and create a new chat training run")


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grounder(checkpoint: dict) -> ml.Grounder:
    """Validate metadata and every parameter before accepting local weights."""
    if (checkpoint.get("architecture") not in {ml.ARCHITECTURE, ARCHITECTURE} or
            checkpoint.get("preprocessing") != ml.PREPROCESSING):
        raise ValueError("The source must be a supported grounding model checkpoint")
    try:
        config = training.validate_config(checkpoint["config"])
        tokenizer = ml.Tokenizer.from_dict(checkpoint["tokenizer"])
        classes = checkpoint["classes"]
        if not isinstance(classes, list) or not classes:
            raise ValueError("The grounding class mapping is missing")
        size = ml.chat_vocabulary_size(checkpoint)
        weights = checkpoint["model"]
        if (not isinstance(weights, dict) or not weights or
                any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                    for value in weights.values())):
            raise ValueError("The checkpoint contains invalid model weights")
        model = ml.Grounder(len(tokenizer.vocabulary), len(classes), config,
                            chat_vocabulary_size=size)
        model.load_state_dict(weights, strict=True)
        return model
    except (KeyError, TypeError, RuntimeError, AttributeError) as error:
        raise ValueError("The checkpoint does not contain a compatible complete grounding model") from error


def _source(payload: dict) -> tuple[dict, str, str | None, str | None]:
    source_run_id, source_chat_run_id = payload.get("source_run_id"), payload.get("source_chat_run_id")
    if bool(source_run_id) == bool(source_chat_run_id):
        raise ValueError("Select exactly one grounding model or grounding + chat model to teach")
    selection = payload.get("source_checkpoint", "latest")
    if not isinstance(selection, str) or not selection:
        raise ValueError("A source checkpoint selection must be a nonempty string")
    if source_run_id:
        _directory(source_run_id)  # Validate IDs before any filesystem access.
        storage.get("run", source_run_id)
        path = training.resolve_checkpoint(source_run_id, selection)
        parent = training.read_checkpoint(path)
    else:
        _directory(source_chat_run_id)
        parent_run = storage.get("chat_run", source_chat_run_id)
        _unified_run(parent_run)
        if selection not in {"latest", "latest.pt"}:
            raise ValueError("Grounding + chat runs support the latest checkpoint")
        path = resolve_checkpoint(source_chat_run_id)
        parent = _read_checkpoint(path, parent_run)
    _grounder(parent)
    return parent, path.name, source_run_id, source_chat_run_id


def sources() -> list[dict]:
    """Discover saved sources cheaply; creation validates the complete checkpoint."""
    available = []
    for kind, records in (("grounding", storage.list_items("run")),
                          ("chat", storage.list_items("chat_run"))):
        for run in records:
            architecture = run.get("architecture", ml.ARCHITECTURE if kind == "grounding" else None)
            if (not run.get("latest_checkpoint") or architecture not in {ml.ARCHITECTURE, ARCHITECTURE} or
                    (kind == "chat" and architecture != ARCHITECTURE)):
                continue
            try:
                if kind == "grounding":
                    training.resolve_checkpoint(run["id"])
                else:
                    if not (_directory(run["id"]) / "checkpoints" / "latest.pt").is_file():
                        continue
            except (ValueError, KeyError, OSError, RuntimeError):
                continue
            capabilities = ["grounding", "chat"] if architecture == ARCHITECTURE else ["grounding"]
            available.append({"id": run["id"], "name": run["name"], "kind": kind,
                              "status": run["status"], "latest_checkpoint": run["latest_checkpoint"],
                              "capabilities": capabilities})
    return available


def create_run(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("A chat run must be an object")
    config = validate_config(payload.get("config", {}))
    examples = storage.list_items("chat_example")
    if not examples:
        raise ValueError("Add at least one chat example before starting training")
    parent, source_checkpoint, source_run_id, source_chat_run_id = _source(payload)
    run_id, timestamp = storage.uid(), storage.now()
    old_characters = list(parent.get("chat_tokenizer", {}).get("characters", []))
    new_characters = {character for example in examples for key in ("prompt", "response")
                      for character in _text(example[key], key)} - set(old_characters)
    tokenizer = {"characters": old_characters + sorted(new_characters), "max_characters": MAX_TEXT}
    # Appending characters keeps every learned token ID stable across teaching runs.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        model = ml.Grounder(len(parent["tokenizer"]["vocabulary"]), len(parent["classes"]),
                            parent["config"], chat_vocabulary_size=len(tokenizer["characters"]) + 4)
    weights = model.state_dict()
    for name, value in parent["model"].items():
        if name in {"chat_embedding.weight", "chat_output.weight", "chat_output.bias"}:
            weights[name][:value.shape[0]].copy_(value)
        else:
            weights[name].copy_(value)
    model.load_state_dict(weights, strict=True)
    source = {"source_run_id": source_run_id, "source_chat_run_id": source_chat_run_id,
              "source_checkpoint": source_checkpoint}
    initial = {"schema": SCHEMA, "kind": "inference_export", "architecture": ARCHITECTURE,
               "preprocessing": ml.PREPROCESSING, "model": model.state_dict(),
               "config": parent["config"], "tokenizer": parent["tokenizer"], "classes": parent["classes"],
               "chat_tokenizer": tokenizer, "chat_training_config": config,
               "capabilities": ["grounding", "chat"], **source}
    initial_path = _directory(run_id) / "initial_weights.pt"
    _atomic_checkpoint(initial_path, initial)
    initial_hash = _file_hash(initial_path)
    snapshot = {"run_id": run_id, "created_at": timestamp, "config": config,
                "architecture": ARCHITECTURE, "preprocessing": ml.PREPROCESSING,
                "grounding_config": parent["config"], "tokenizer": parent["tokenizer"],
                "classes": parent["classes"], "chat_tokenizer": tokenizer, "examples": examples,
                "initial_weights_sha256": initial_hash, **source}
    fingerprint = _fingerprint(snapshot)
    _json_atomic(_directory(run_id) / "experiment.json", snapshot)
    run = {"id": run_id, "name": str(payload.get("name") or f"Chat {run_id[:8]}").strip()[:160],
           "created_at": timestamp, "config": config, "status": "created",
           "example_count": len(examples), "snapshot_fingerprint": fingerprint,
           "grounding_config": parent["config"], "capabilities": ["grounding", "chat"],
           "initial_weights_sha256": initial_hash, **source,
           "architecture": ARCHITECTURE, "latest_checkpoint": None, "pid": None,
           "requested_control": None, "error": None,
           "progress": {"epoch": 0, "global_step": 0, "loss": None, "elapsed_seconds": 0.0}}
    return storage.put("chat_run", run_id, run)


def _snapshot(run: dict) -> dict:
    _unified_run(run)
    try:
        snapshot = json.loads((_directory(run["id"]) / "experiment.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("The chat run's immutable teaching snapshot is missing or unreadable") from error
    if (_fingerprint(snapshot) != run["snapshot_fingerprint"] or
            snapshot.get("run_id") != run["id"] or snapshot.get("config") != run["config"] or
            snapshot.get("architecture") != ARCHITECTURE or
            snapshot.get("grounding_config") != run.get("grounding_config") or
            any(snapshot.get(key) != run.get(key) for key in
                ("source_run_id", "source_chat_run_id", "source_checkpoint", "initial_weights_sha256")) or
            len(snapshot.get("examples", [])) != run["example_count"]):
        raise ValueError("The chat run's immutable experiment changed; create a new run")
    validate_config(snapshot["config"])
    try:
        unchanged = _file_hash(_directory(run["id"]) / "initial_weights.pt") == snapshot["initial_weights_sha256"]
    except OSError as error:
        raise ValueError("The chat run's immutable initial weights are missing or unreadable") from error
    if not unchanged:
        raise ValueError("The chat run's immutable initial weights changed; create a new run")
    return snapshot


def _lookup(tokenizer: dict) -> dict:
    if not isinstance(tokenizer, dict):
        raise ValueError("Invalid chat checkpoint tokenizer")
    characters = tokenizer.get("characters")
    if (not isinstance(characters, list) or not characters or
            any(not isinstance(char, str) or len(char) != 1 for char in characters) or
            len(characters) != len(set(characters)) or tokenizer.get("max_characters") != MAX_TEXT):
        raise ValueError("Invalid chat checkpoint tokenizer")
    return {char: index + 4 for index, char in enumerate(characters)}


def _load_checkpoint(path: Path) -> dict:
    # Only application-created local files are loadable, and pickle globals are disabled.
    try:
        deadline = time.monotonic() + 3.0
        while True:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                break
            except PermissionError:
                if os.name != "nt" or time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
    except Exception as error:
        raise ValueError("The chat checkpoint could not be read") from error
    if not isinstance(checkpoint, dict):
        raise ValueError("The chat checkpoint must contain a model record")
    return checkpoint


def _read_checkpoint(path: Path, run: dict) -> dict:
    _unified_run(run)
    checkpoint = _load_checkpoint(path)
    if (not isinstance(checkpoint, dict) or checkpoint.get("schema") != SCHEMA or
            checkpoint.get("kind") != "grounding_chat_training_checkpoint" or
            checkpoint.get("architecture") != ARCHITECTURE or checkpoint.get("run_id") != run["id"] or
            checkpoint.get("snapshot_fingerprint") != run["snapshot_fingerprint"] or
            checkpoint.get("config") != run.get("grounding_config") or
            checkpoint.get("chat_training_config") != run["config"] or
            checkpoint.get("initial_weights_sha256") != run.get("initial_weights_sha256") or
            checkpoint.get("preprocessing") != ml.PREPROCESSING):
        raise ValueError("Unsupported or incompatible chat checkpoint")
    for key in ("model", "optimizer", "tokenizer", "chat_tokenizer", "classes", "epoch", "cursor", "global_step", "loss"):
        if key not in checkpoint:
            raise ValueError(f"Chat checkpoint is incomplete: missing {key}")
    _lookup(checkpoint["chat_tokenizer"])
    if (not isinstance(checkpoint["model"], dict) or not checkpoint["model"] or
            any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all()
                for value in checkpoint["model"].values())):
        raise ValueError("Chat checkpoint contains invalid model weights")
    for key in ("epoch", "cursor", "global_step"):
        if isinstance(checkpoint[key], bool) or not isinstance(checkpoint[key], int) or checkpoint[key] < 0:
            raise ValueError(f"Invalid checkpoint {key}")
    if checkpoint["epoch"] > run["config"]["epochs"] or checkpoint["cursor"] > run["example_count"]:
        raise ValueError("Invalid chat checkpoint progress")
    if checkpoint["loss"] is not None and (isinstance(checkpoint["loss"], bool) or
            not isinstance(checkpoint["loss"], (int, float)) or not math.isfinite(checkpoint["loss"])):
        raise ValueError("Invalid chat checkpoint loss")
    return checkpoint


def resolve_checkpoint(run_id: str) -> Path:
    run = storage.get("chat_run", run_id)
    _unified_run(run)
    path = _directory(run_id) / "checkpoints" / "latest.pt"
    if not path.is_file():
        raise ValueError("Train this chat run before testing or exporting it")
    if _read_checkpoint(path, run)["global_step"] < 1:
        raise ValueError("This chat run has not completed a learning update yet")
    return path


def _worker_alive(run: dict) -> bool:
    pid = run.get("pid")
    if not pid:
        return False
    try:
        process = psutil.Process(pid)
        arguments = process.cmdline()
        if "grounding.chat" not in arguments or "--run" not in arguments:
            return False
        run_position = arguments.index("--run") + 1
        if run_position >= len(arguments) or arguments[run_position] != run["id"]:
            return False
        return process.is_running() and (run.get("pid_created_at") is None or
               abs(process.create_time() - run["pid_created_at"]) < 0.01)
    except (psutil.Error, OSError):
        return False


def _reserved(run: dict) -> bool:
    if run["status"] not in {"running", "queued"}:
        return False
    if _worker_alive(run):
        return True
    if run["status"] == "queued" and not run.get("pid"):
        created = datetime.fromisoformat(run.get("launch_requested_at") or run["created_at"])
        return (datetime.now(timezone.utc) - created).total_seconds() < 60
    return False


def recover_runs() -> None:
    with LAUNCH_LOCK:
        for run in storage.list_items("chat_run"):
            if run["status"] in {"running", "queued"} and not _reserved(run):
                _update(run["id"], status="interrupted", pid=None, pid_created_at=None,
                        error="Chat worker stopped unexpectedly. Resume from the latest saved update.")


def launch_run(run_id: str) -> dict:
    with LAUNCH_LOCK:
        for grounding_run in storage.list_items("run"):
            if grounding_run["status"] in {"running", "queued"} and training._worker_alive(grounding_run):
                raise ValueError("A grounding training worker is active. Stop it before starting chat training")
        run = storage.get("chat_run", run_id)
        if run["status"] == "completed":
            raise ValueError("This chat run completed its epochs. Create a new run to train again")
        snapshot = _snapshot(run)
        existing = _directory(run_id) / "checkpoints" / "latest.pt"
        if existing.exists():
            checkpoint = _read_checkpoint(existing, run)
            if any(checkpoint[key] != snapshot[key] for key in ("tokenizer", "chat_tokenizer", "classes")):
                raise ValueError("The chat checkpoint tokenizer does not match its teaching snapshot")
            _grounder(checkpoint)
        # A database reservation prevents two API processes from launching together.
        with storage.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute("SELECT id,data FROM documents WHERE kind='chat_run'").fetchall()
            for other_id, data in rows:
                other = json.loads(data)
                if _reserved(other):
                    raise ValueError("A chat training worker is already active. Stop it before starting another run")
                if other_id == run_id:
                    run = other
            run.update(status="queued", error=None, requested_control=None, pid=None,
                       pid_created_at=None, launch_requested_at=storage.now())
            conn.execute("UPDATE documents SET data=? WHERE kind='chat_run' AND id=?",
                         (json.dumps(run, allow_nan=False), run_id))
        (_directory(run_id) / "control.json").unlink(missing_ok=True)
        env = os.environ.copy()
        env["GROUNDING_DATA_DIR"] = str(storage.ROOT)
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            with (_directory(run_id) / "worker.log").open("ab") as log:
                process = subprocess.Popen([sys.executable, "-m", "grounding.chat", "--run", run_id],
                                           cwd=str(Path(__file__).resolve().parents[1]), env=env,
                                           stdout=log, stderr=subprocess.STDOUT,
                                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            PROCESSES[run_id] = process
            try:
                pid_created_at = psutil.Process(process.pid).create_time()
            except psutil.Error:
                pid_created_at = None
            # A fast worker may have finished already; do not restore its cleared PID.
            with storage.connection() as conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute("SELECT data FROM documents WHERE kind='chat_run' AND id=?", (run_id,)).fetchone()
                current = json.loads(row[0])
                if current["status"] == "queued":
                    current.update(pid=process.pid, pid_created_at=pid_created_at)
                    conn.execute("UPDATE documents SET data=? WHERE kind='chat_run' AND id=?",
                                 (json.dumps(current, allow_nan=False), run_id))
            return current
        except Exception as error:
            _update(run_id, status="error", error=str(error), pid=None, pid_created_at=None)
            raise


def control_run(run_id: str, action: str) -> dict:
    if action == "resume":
        return launch_run(run_id)
    if action != "stop":
        raise ValueError("Chat training action must be stop or resume")
    with LAUNCH_LOCK:
        run = storage.get("chat_run", run_id)
        if run["status"] == "stopped":
            return run
        if run["status"] in {"created", "interrupted", "error"}:
            return _update(run_id, status="stopped", requested_control=None)
        if run["status"] not in {"running", "queued"}:
            raise ValueError(f"Cannot stop a {run['status']} chat run")
        _json_atomic(_directory(run_id) / "control.json", {"action": "stop", "requested_at": storage.now()})
        return _update(run_id, requested_control="stop")


def _stop_requested(run_id: str) -> bool:
    path = _directory(run_id) / "control.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("action") == "stop"


def _batch(examples: list[dict], lookup: dict, device):
    def tokens(text):
        return [lookup.get(character, UNK) for character in text]
    prompts = [torch.tensor(tokens(row["prompt"]) + [EOS], dtype=torch.long) for row in examples]
    inputs = [torch.tensor([BOS] + tokens(row["response"]), dtype=torch.long) for row in examples]
    targets = [torch.tensor(tokens(row["response"]) + [EOS], dtype=torch.long) for row in examples]
    lengths = torch.tensor([len(prompt) for prompt in prompts])
    return (pad_sequence(prompts, batch_first=True).to(device), lengths,
            pad_sequence(inputs, batch_first=True).to(device),
            pad_sequence(targets, batch_first=True).to(device))


def run_worker(run_id: str, *, stop_after_epochs: int | None = None) -> dict:
    """Train and checkpoint at epoch/stop boundaries; the limit supports restart tests."""
    storage.init_db()
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    run = storage.get("chat_run", run_id)
    if run["status"] == "completed":
        raise ValueError("This chat run has already completed")
    started = time.perf_counter()
    previous_elapsed = run["progress"].get("elapsed_seconds", 0.0)
    try:
        snapshot = _snapshot(run)
        config, tokenizer = run["config"], snapshot["chat_tokenizer"]
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable. Create a CPU chat run")
        device = torch.device("cuda" if config["device"] != "cpu" and torch.cuda.is_available() else "cpu")
        torch.manual_seed(config["seed"])
        initial = _load_checkpoint(_directory(run_id) / "initial_weights.pt")
        if (initial.get("config") != snapshot["grounding_config"] or
                any(initial.get(key) != snapshot[key] for key in ("tokenizer", "chat_tokenizer", "classes"))):
            raise ValueError("The initial model does not match the immutable teaching snapshot")
        model = _grounder(initial).to(device)
        # The shared text encoder still propagates gradients into chat embeddings.
        # Its weights and all visual heads remain identical to the chosen parent.
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith(("chat_embedding.", "chat_decoder.", "chat_output.")))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=config["learning_rate"])
        examples, lookup = snapshot["examples"], _lookup(tokenizer)
        epoch, cursor, global_step, loss_value = 0, 0, 0, None
        checkpoint_path = _directory(run_id) / "checkpoints" / "latest.pt"
        if checkpoint_path.exists():
            checkpoint = _read_checkpoint(checkpoint_path, run)
            if (checkpoint["chat_tokenizer"] != tokenizer or
                    any(checkpoint[key] != snapshot[key] for key in ("tokenizer", "classes"))):
                raise ValueError("The checkpoint tokenizer changed")
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            epoch, cursor, global_step, loss_value = (checkpoint[key] for key in ("epoch", "cursor", "global_step", "loss"))
        _update(run_id, status="running", error=None, pid=os.getpid(),
                pid_created_at=psutil.Process(os.getpid()).create_time(), device=str(device))

        def progress():
            return {"epoch": epoch, "global_step": global_step, "loss": loss_value,
                    "elapsed_seconds": previous_elapsed + time.perf_counter() - started}

        def save():
            state = {"schema": SCHEMA, "kind": "grounding_chat_training_checkpoint", "architecture": ARCHITECTURE,
                     "preprocessing": ml.PREPROCESSING, "capabilities": ["grounding", "chat"],
                     "run_id": run_id, "config": snapshot["grounding_config"], "chat_training_config": config,
                     "snapshot_fingerprint": run["snapshot_fingerprint"],
                     "initial_weights_sha256": run["initial_weights_sha256"],
                     "tokenizer": snapshot["tokenizer"], "classes": snapshot["classes"], "chat_tokenizer": tokenizer,
                     **{key: run[key] for key in ("source_run_id", "source_chat_run_id", "source_checkpoint")},
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "cursor": cursor, "global_step": global_step, "loss": loss_value}
            _atomic_checkpoint(checkpoint_path, state)
            _update(run_id, latest_checkpoint="latest.pt" if global_step else None, progress=progress())

        initial_epoch = epoch
        model.train()
        while epoch < config["epochs"]:
            if _stop_requested(run_id):
                save()
                return _update(run_id, status="stopped", requested_control=None, pid=None, pid_created_at=None)
            order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(config["seed"] + epoch)).tolist()
            while cursor < len(order):
                rows = [examples[index] for index in order[cursor:cursor + config["batch_size"]]]
                prompts, lengths, inputs, targets = _batch(rows, lookup, device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.chat_forward(prompts, lengths, inputs)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=PAD)
                if not torch.isfinite(loss):
                    raise ValueError("Chat loss became nonfinite. Create a new run with a lower learning rate")
                loss.backward()
                nn.utils.clip_grad_norm_(trainable, 5.0, error_if_nonfinite=True)
                optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise ValueError("Chat weights became nonfinite. The previous checkpoint was preserved")
                global_step += 1
                cursor += len(rows)
                loss_value = float(loss.detach())
                if _stop_requested(run_id):
                    if cursor == len(order):
                        epoch, cursor = epoch + 1, 0
                    save()
                    return _update(run_id, status="stopped", requested_control=None, pid=None, pid_created_at=None)
            epoch, cursor = epoch + 1, 0
            save()
            if stop_after_epochs is not None and epoch - initial_epoch >= stop_after_epochs and epoch < config["epochs"]:
                return _update(run_id, status="stopped", pid=None, pid_created_at=None, requested_control=None)
        return _update(run_id, status="completed", progress=progress(), pid=None,
                       pid_created_at=None, requested_control=None, completed_at=storage.now())
    except Exception as error:
        _update(run_id, status="error", error=str(error), pid=None, pid_created_at=None,
                requested_control=None)
        raise


def predict(run_id: str, message: str) -> dict:
    message = _text(message, "Message")
    run = storage.get("chat_run", run_id)
    path = resolve_checkpoint(run_id)
    checkpoint = _read_checkpoint(path, run)
    lookup = _lookup(checkpoint["chat_tokenizer"])
    model = _grounder(checkpoint)
    model.eval()
    prompt = torch.tensor([[lookup.get(character, UNK) for character in message] + [EOS]])
    reply = []
    with torch.inference_mode():
        hidden = model.encode_chat(prompt, torch.tensor([prompt.shape[1]]))
        previous = torch.tensor([[BOS]])
        for _ in range(MAX_TEXT):
            scores, hidden = model.chat_decode_step(previous, hidden)
            scores[:, [PAD, BOS, UNK]] = -torch.inf
            token = int(scores.argmax(dim=-1))
            if token == EOS:
                break
            reply.append(checkpoint["chat_tokenizer"]["characters"][token - 4])
            previous = torch.tensor([[token]])
    return {"reply": "".join(reply), "run_id": run_id, "checkpoint": path.name,
            "unknown_characters": sum(character not in lookup for character in message)}


def export_model(run_id: str) -> dict:
    run = storage.get("chat_run", run_id)
    checkpoint = _read_checkpoint(resolve_checkpoint(run_id), run)
    exported = {key: checkpoint[key] for key in (
        "schema", "architecture", "preprocessing", "run_id", "config", "tokenizer", "classes", "model", "global_step",
        "chat_tokenizer", "chat_training_config", "capabilities", "source_run_id", "source_chat_run_id", "source_checkpoint")}
    exported.update(kind="inference_export", created_at=storage.now())
    # Each download receives an immutable file, including downloads while the
    # same run is learning. A slow Windows download cannot lock the next export.
    export_id = storage.uid()
    relative_path = f"chat/runs/{run_id}/exports/{export_id}.pt"
    _atomic_checkpoint(storage.safe_path(relative_path), exported)
    return storage.put("export", export_id, {"id": export_id, "run_id": run_id,
                       "checkpoint": "latest.pt", "path": relative_path, "kind": "inference_export",
                       "architecture": ARCHITECTURE, "capabilities": ["grounding", "chat"],
                       "created_at": exported["created_at"],
                       "download_url": f"/api/exports/{export_id}/download"})


def export_path(run_id: str) -> Path:
    return storage.safe_path(export_model(run_id)["path"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Teach chat replies to an existing local grounding model")
    parser.add_argument("--run", required=True)
    arguments = parser.parse_args()
    run_worker(arguments.run)


if __name__ == "__main__":
    main()

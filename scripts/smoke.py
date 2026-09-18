r"""Real subprocess workflow verification using synthetic data, never a quality benchmark.

Run: .venv\Scripts\python.exe scripts\smoke.py
The web server and every training worker are real subprocesses. Data is isolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
import psutil

REPOSITORY = Path(__file__).resolve().parents[1]
TERMINAL = {"completed", "paused", "stopped", "error", "interrupted"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, help="Keep verification artifacts in a new or empty folder.")
    parser.add_argument("--timeout", type=float, default=180, help="Maximum seconds per worker phase.")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Exercise the laptop GPU by default; CPU remains available.")
    args = parser.parse_args()
    temporary = None
    if args.data_dir:
        root = args.data_dir.resolve()
        if root.exists() and any(root.iterdir()):
            raise SystemExit("--data-dir must be a new or empty directory; existing data is never reused.")
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="groundwork-smoke-")
        root = Path(temporary.name)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    env = dict(os.environ, GROUNDING_DATA_DIR=str(root), PYTHONUNBUFFERED="1")
    client = httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120, trust_env=False)
    server: subprocess.Popen | None = None
    server_log = (root / "smoke-server.log").open("a", encoding="utf-8")
    run_ids: list[str] = []
    worker_processes: dict[int, psutil.Process] = {}
    started = time.monotonic()
    evidence: dict = {"kind": "synthetic workflow verification", "quality_claim": False, "device": args.device, "stages": []}

    def stage(message: str) -> None:
        print(message, flush=True)
        evidence["stages"].append(message)

    def request(method: str, path: str, **kwargs):
        response = client.request(method, "/api" + path, **kwargs)
        if response.is_error:
            raise AssertionError(f"{method} {path}: {response.status_code} {response.text}")
        result = response.json()
        if isinstance(result, dict) and result.get("pid") and "config" in result and "version_id" in result:
            try:
                worker_processes.setdefault(result["pid"], psutil.Process(result["pid"]))
            except psutil.NoSuchProcess:
                pass
        return result

    def start_server() -> None:
        nonlocal server
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "grounding.api:app", "--host", "127.0.0.1", "--port", str(port), "--no-access-log"],
            cwd=REPOSITORY, env=env, stdout=server_log, stderr=subprocess.STDOUT, creationflags=flags,
        )
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise AssertionError(f"Server exited {server.returncode}; see {root / 'smoke-server.log'}")
            try:
                if client.get("/api/health", timeout=1).is_success:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.1)
        raise AssertionError("Server startup timed out")

    def stop_server() -> None:
        nonlocal server
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
        server = None

    def wait_run(run_id: str, predicate, label: str) -> dict:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            run = request("GET", f"/runs/{run_id}")
            if run["status"] == "error":
                raise AssertionError(f"{label}: training failed: {run.get('error')}")
            if predicate(run):
                return run
            time.sleep(0.1)
        raise AssertionError(f"{label}: timed out after {args.timeout}s; latest run: {run}")

    def checkpoint_path(run: dict) -> Path:
        name = run["latest_checkpoint"]
        supplied = Path(name)
        candidates = [supplied, root / supplied, root / "runs" / run["id"] / supplied,
                      root / "runs" / run["id"] / "checkpoints" / supplied]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        raise AssertionError(f"Cannot find checkpoint {name!r}")

    try:
        start_server()
        stage("Started isolated local API subprocess")
        evidence["hardware"] = request("GET", "/dashboard")["gpu"]
        generated = request("POST", "/synthetic", json={"count": 18, "seed": 123, "reviewed": True})
        assert generated["images"] == 18 and generated["examples"] > 0
        validation = request("POST", "/datasets/validate", json={"group_by": "group", "seed": 123})
        assert not validation["errors"], validation
        version = request("POST", "/versions", json={"name": "Synthetic smoke v1", "seed": 123, "group_by": "group"})
        manifest = request("GET", f"/versions/{version['id']}")
        assert manifest["records"] and {r["split"] for r in manifest["records"]} >= {"train"}
        stage("Generated, explicitly reviewed, validated, and versioned synthetic examples")
        config = {"image_size": 64, "batch_size": 2, "epochs": 3, "learning_rate": 0.001,
                  "seed": 123, "grad_accum": 2, "width": 8, "text_dim": 16, "max_tokens": 24,
                  "device": args.device, "mixed_precision": args.device == "cuda", "checkpoint_every": 1}
        run = request("POST", "/runs", json={"name": "Smoke fresh", "version_id": version["id"], "mode": "fresh", "config": config})
        run_ids.append(run["id"])
        progressed = wait_run(run["id"], lambda r: bool(r.get("latest_checkpoint")) and r.get("progress", {}).get("global_step", 0) >= 1, "first checkpoint")
        assert progressed["status"] != "completed", "Tiny run finished before pause; increase smoke epochs"
        request("POST", f"/runs/{run['id']}/pause")
        paused = wait_run(run["id"], lambda r: r["status"] == "paused", "pause")
        paused_step = paused["progress"]["global_step"]
        assert checkpoint_path(paused).is_file()
        stage(f"Paused real training worker at optimizer step {paused_step} with checkpoint")
        stop_server()
        start_server()
        persisted = request("GET", f"/runs/{run['id']}")
        assert persisted["status"] == "paused" and persisted["progress"]["global_step"] == paused_step
        request("POST", f"/runs/{run['id']}/resume")
        completed = wait_run(run["id"], lambda r: r["status"] == "completed", "resume completion")
        assert completed["progress"]["global_step"] > paused_step
        if args.device == "cuda":
            assert completed.get("memory_probe", {}).get("passed"), "CUDA worker must report a successful full-update probe"
        evidence["memory_probe"] = completed.get("memory_probe")
        evidence["training_memory"] = completed["progress"].get("memory")
        stage(f"Restarted API, resumed a new training process, completed at step {completed['progress']['global_step']}")
        sample = manifest["records"][0]
        sample_path = root / sample["image_path"]
        with sample_path.open("rb") as image_file:
            prediction = request("POST", "/predict", files={"file": (sample_path.name, image_file, "image/png")},
                                 data={"instruction": sample["instruction"], "run_id": run["id"], "checkpoint": "latest", "threshold": "0"})
        assert prediction["target_present"] and prediction["bbox"] is not None
        x1, y1, x2, y2 = prediction["bbox"]
        assert 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        assert 0 <= prediction["presence_score"] <= 1
        assert prediction["width"] == sample["width"] and prediction["height"] == sample["height"]
        with sample_path.open("rb") as image_file:
            abstention = request("POST", "/predict", files={"file": (sample_path.name, image_file, "image/png")},
                                 data={"instruction": sample["instruction"], "run_id": run["id"], "checkpoint": "latest", "threshold": "1"})
        assert not abstention["target_present"] and abstention["bbox"] is None and abstention["click_point"] is None
        request("POST", "/predictions/correct", json={"image_id": prediction["image_id"], "instruction": sample["instruction"],
                                                       "target_present": True, "class_id": prediction["class_id"],
                                                       "bbox": prediction["bbox"], "click_point": prediction["click_point"]})
        corrected_image = request("GET", f"/images/{prediction['image_id']}")
        assert corrected_image["examples"][-1]["status"] == "draft"
        split = next((value for value in ("test", "val", "train") if any(r["split"] == value for r in manifest["records"])))
        evaluation = request("POST", "/evaluate", json={"run_id": run["id"], "checkpoint": "latest", "version_id": version["id"], "split": split, "threshold": 0.5})
        assert evaluation["examples"] and evaluation["metrics"] and evaluation["baselines"]
        exported = request("POST", f"/runs/{run['id']}/export", json={"checkpoint": "latest"})
        download = client.get(exported.get("download_url", f"/api/exports/{exported['id']}/download"))
        assert download.is_success and len(download.content) > 100
        stage("Verified valid-box inference, abstention, draft correction, evaluation, and downloadable inference export")
        parent_path = checkpoint_path(completed)
        parent_hash = digest(parent_path)
        parent_record = request("GET", f"/runs/{run['id']}")
        request("POST", "/synthetic", json={"count": 6, "seed": 456, "reviewed": True})
        new_version = request("POST", "/versions", json={"name": "Synthetic smoke v2 with added data", "seed": 123, "group_by": "group"})
        retrain_config = dict(config, epochs=1)
        retrain = request("POST", "/runs", json={"name": "Smoke retrain", "version_id": new_version["id"], "mode": "retrain",
                                                  "parent_run_id": run["id"], "source_checkpoint": "latest", "config": retrain_config})
        run_ids.append(retrain["id"])
        retrained = wait_run(retrain["id"], lambda r: r["status"] == "completed", "retraining")
        assert retrained["id"] != run["id"] and retrained["parent_run_id"] == run["id"]
        assert retrained["version_id"] == new_version["id"]
        assert digest(parent_path) == parent_hash
        assert request("GET", f"/runs/{run['id']}") == parent_record
        stage("Trained a separate child on old plus new synthetic data; verified parent bytes and run history unchanged")
        export_child = request("POST", "/runs", json={"name": "Smoke export retrain", "version_id": new_version["id"], "mode": "retrain",
                                                       "source_export_id": exported["id"], "config": retrain_config})
        run_ids.append(export_child["id"])
        export_retrained = wait_run(export_child["id"], lambda r: r["status"] == "completed", "export retraining")
        assert export_retrained["id"] != run["id"] and export_retrained["mode"] == "retrain"
        assert digest(parent_path) == parent_hash
        stage("Created and completed a fresh retraining run from inference-export weights")
        evidence.update({"passed": True, "elapsed_seconds": round(time.monotonic() - started, 2),
                         "fresh_run_id": run["id"], "paused_step": paused_step,
                         "completed_step": completed["progress"]["global_step"],
                         "retrain_run_id": retrain["id"], "export_retrain_run_id": export_child["id"],
                         "parent_checkpoint_sha256": parent_hash})
        (root / "smoke-result.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        print(json.dumps(evidence, indent=2), flush=True)
        if args.data_dir:
            print(f"Artifacts: {root}", flush=True)
    except Exception:
        server_log.flush()
        log_path = root / "smoke-server.log"
        if log_path.exists():
            print(log_path.read_text(encoding="utf-8", errors="replace")[-12000:], file=sys.stderr)
        for run_id in run_ids:
            worker_log = root / "runs" / run_id / "worker.log"
            if worker_log.exists():
                print(f"Worker {run_id}:\n" + worker_log.read_text(encoding="utf-8", errors="replace")[-6000:], file=sys.stderr)
        raise
    finally:
        for run_id in run_ids:
            try:
                current = request("GET", f"/runs/{run_id}")
                if current["status"] not in TERMINAL:
                    request("POST", f"/runs/{run_id}/stop")
                    wait_run(run_id, lambda r: r["status"] in TERMINAL, "cleanup")
            except Exception:
                pass
        stop_server()
        # Windows keeps inherited worker logs open until the actual process exits,
        # briefly after the final status is committed. Wait before removing temp data.
        for process in worker_processes.values():
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        client.close()
        server_log.close()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()

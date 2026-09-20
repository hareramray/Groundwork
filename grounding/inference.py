"""Shared preprocessing, honest abstention-aware metrics, and measured baselines."""
from __future__ import annotations

import os
import statistics
import threading
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps

from . import storage
from .dataset import verify_version
from .ml import SUPPORTED_ARCHITECTURES, PREPROCESSING, Grounder, Tokenizer, chat_vocabulary_size, box_iou_giou, make_batch, preprocess_image
from .training import read_checkpoint, resolve_checkpoint, select_device

INFERENCE_LOCK = threading.Lock()


def process_memory_mb() -> float:
    """Current process working set/RSS; GPU allocations are reported separately."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        memory_info.restype = wintypes.BOOL
        if not memory_info(handle, ctypes.byref(counters), counters.cb):
            raise OSError("Unable to measure the current process working set")
        return counters.WorkingSetSize / 2**20
    status = Path("/proc/self/statm")
    if status.exists():
        return int(status.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (2**20 if os.sys.platform == "darwin" else 1024)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _check_threshold(threshold: float) -> float:
    if not 0 <= threshold <= 1:
        raise ValueError("Presence threshold must be between zero and one")
    return float(threshold)


def load_model(run_id: str, selection: str = "latest"):
    path = resolve_checkpoint(run_id, selection)
    checkpoint = read_checkpoint(path)
    if checkpoint.get("architecture") not in SUPPORTED_ARCHITECTURES or checkpoint.get("preprocessing") != PREPROCESSING:
        raise ValueError("This model uses an unsupported architecture or preprocessing format")
    inference_config = dict(checkpoint["config"])
    if not torch.cuda.is_available():
        inference_config["device"] = "cpu"
    device = select_device(inference_config)
    tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])
    model = Grounder(len(tokenizer.vocabulary), len(checkpoint["classes"]), checkpoint["config"],
                     chat_vocabulary_size(checkpoint)).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, tokenizer, checkpoint, device, path


def format_prediction(output: dict, index: int, width: int, height: int, classes: list[str],
                      threshold: float, instruction: str, latency_ms: float, memory_mb: float) -> dict:
    score = float(output["presence_logits"][index].float().sigmoid().detach().cpu())
    bbox = output["bbox"][index].float().detach().cpu().tolist()
    class_id = int(output["class_logits"][index].argmax().detach().cpu())
    present = score >= threshold
    click = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    return {"target_present": present, "presence_score": score, "threshold": threshold,
            "class_id": class_id if present else None, "class_name": classes[class_id] if present else None,
            "bbox": bbox if present else None, "click_point": click if present else None,
            "bbox_pixels": [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height] if present else None,
            "click_point_pixels": [click[0] * width, click[1] * height] if present else None,
            "raw_bbox": bbox, "raw_class_id": class_id, "width": width, "height": height,
            "instruction": instruction, "latency_ms": latency_ms, "memory_mb": memory_mb,
            "message": "Candidate target prediction" if present else "No confident target prediction",
            "score_description": "Target-presence score; not a calibrated click-success probability"}


def predict(run_id: str, checkpoint: str, image_path: str | Path, instruction: str, threshold: float = 0.5) -> dict:
    threshold = _check_threshold(threshold)
    if not instruction.strip():
        raise ValueError("An instruction is required")
    with INFERENCE_LOCK:
        model, tokenizer, state, device, path = load_model(run_id, checkpoint)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started = time.perf_counter()
        with Image.open(image_path) as opened:
            oriented = ImageOps.exif_transpose(opened)
            width, height = oriented.size
            images = preprocess_image(oriented, state["config"]["image_size"]).unsqueeze(0).to(device)
        ids, length = tokenizer.encode(instruction)
        with torch.inference_mode():
            output = model(images, torch.tensor([ids], device=device), torch.tensor([length], device=device))
        _synchronize(device)
        latency = (time.perf_counter() - started) * 1000
        gpu_memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
        rss = process_memory_mb()
        result = format_prediction(output, 0, width, height, state["classes"], threshold, instruction,
                                   latency, gpu_memory if device.type == "cuda" else rss)
        result.update(run_id=run_id, checkpoint=path.name, device=str(device),
                      preprocessing=state["preprocessing"], image_size=state["config"]["image_size"],
                      gpu_peak_allocated_mb=gpu_memory, process_rss_mb=rss,
                      latency_scope="image decode, preprocessing, device transfer, and forward; excludes model loading",
                      unknown_words=sorted({word for word in tokenizer.words(instruction) if word not in tokenizer.lookup}))
        return result


def calculate_metrics(records: list[dict], predictions: list[dict]) -> tuple[dict, list[dict]]:
    positives = sum(bool(row["target_present"]) for row in records)
    negatives = len(records) - positives
    tp = fp = attempted = successes = class_correct = 0
    ious, details = [], []
    class_totals, class_hits = Counter(), Counter()
    for row, prediction in zip(records, predictions):
        attempted += bool(prediction["target_present"])
        iou, success = None, False
        if row["target_present"]:
            tp += bool(prediction["target_present"])
            iou = float(box_iou_giou(torch.tensor([prediction["raw_bbox"]]), torch.tensor([row["bbox"]]))[0][0])
            ious.append(iou)
            correct = prediction["raw_class_id"] == row["class_id"]
            class_correct += correct
            class_totals[row["class_id"]] += 1
            class_hits[row["class_id"]] += correct
            success = bool(prediction["target_present"] and correct and iou >= 0.5)
            successes += success
        else:
            fp += bool(prediction["target_present"])
        details.append({**row, "prediction": prediction,
                        "success": success if row["target_present"] else not prediction["target_present"],
                        "positive_grounding_success": success if row["target_present"] else None,
                        "correct_absence": not row["target_present"] and not prediction["target_present"],
                        "iou": iou, "image_url": f"/api/files/{row['image_path']}"})
    ratio = lambda a, b: a / b if b else None
    metrics = {"examples": len(records), "positive_examples": positives, "absent_examples": negatives,
               "grounding_success": ratio(successes, positives), "grounding_success_count": successes,
               "mean_iou": statistics.mean(ious) if ious else None,
               "class_accuracy": ratio(class_correct, positives),
               "per_class_accuracy": {str(key): ratio(class_hits[key], value) for key, value in sorted(class_totals.items())},
               "presence_precision": ratio(tp, tp + fp), "presence_recall": ratio(tp, positives),
               "presence_accuracy": ratio(tp + negatives - fp, len(records)),
               "false_positives_absent": fp, "false_positive_rate_absent": ratio(fp, negatives),
               "coverage": ratio(attempted, len(records)), "attempted_predictions": attempted,
               "selective_correctness": ratio(successes, attempted),
               "positive_abstentions": positives - tp,
               "latency_ms_mean": statistics.mean([p["latency_ms"] for p in predictions]) if predictions else None,
               "latency_ms_p50": statistics.median([p["latency_ms"] for p in predictions]) if predictions else None,
               "definitions": {"grounding_success": "Positive target predicted present, IoU >= 0.5, and class correct. Every positive abstention fails.",
                               "mean_iou": "Raw predicted box IoU on every positive example, including abstentions.",
                               "class_accuracy": "Raw class-head accuracy on every positive example, including abstentions.",
                               "coverage": "Predictions above presence threshold / all examples.",
                               "selective_correctness": "Successful positive groundings / all attempted predictions; absent false positives fail.",
                               "latency": "Amortized per-example preprocessing and forward time measured per batch; excludes model loading."}}
    return metrics, details


def _predictions(model: Grounder, tokenizer: Tokenizer, config: dict, classes: list,
                 records: list[dict], device: torch.device, threshold: float) -> list[dict]:
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), config["batch_size"]):
            rows = records[start:start + config["batch_size"]]
            _synchronize(device)
            begun = time.perf_counter()
            batch = make_batch(rows, tokenizer, config, storage.ROOT, device)
            output = model(batch["images"], batch["tokens"], batch["lengths"])
            _synchronize(device)
            elapsed = (time.perf_counter() - begun) * 1000 / len(rows)
            memory = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else process_memory_mb()
            for i, row in enumerate(rows):
                predictions.append(format_prediction(output, i, row["width"], row["height"], classes,
                                                      threshold, row["instruction"], elapsed, memory))
    return predictions


def evaluate_records(model: Grounder, tokenizer: Tokenizer, config: dict, classes: list,
                     records: list[dict], device: torch.device, threshold: float = 0.5,
                     include_examples: bool = True, baselines: bool = True,
                     baseline_records: list[dict] | None = None) -> tuple[dict, dict, list[dict]]:
    threshold = _check_threshold(threshold)
    predictions = _predictions(model, tokenizer, config, classes, records, device, threshold)
    metrics, examples = calculate_metrics(records, predictions)
    metrics["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    metrics["process_rss_mb"] = process_memory_mb()
    metrics["device"] = str(device)
    baseline_metrics = {}
    if baselines and records:
        if len({row["instruction"] for row in records}) > 1:
            shifted = [{**row, "instruction": records[(i + 1) % len(records)]["instruction"]} for i, row in enumerate(records)]
            shuffled_predictions = _predictions(model, tokenizer, config, classes, shifted, device, threshold)
            baseline_metrics["shifted_instructions"] = calculate_metrics(records, shuffled_predictions)[0]
            baseline_metrics["shifted_instructions"]["description"] = (
                "Instructions cyclically shifted one example; image and target remain fixed. Similar results can suggest instruction neglect. Some repeated instructions may remain unchanged.")
        else:
            baseline_metrics["shifted_instructions"] = {"available": False, "reason": "At least two distinct instructions are required"}
        positive_train = [row for row in (baseline_records or []) if row["target_present"]]
        if positive_train:
            mean_box = [sum(row["bbox"][i] for row in positive_train) / len(positive_train) for i in range(4)]
            majority = Counter(row["class_id"] for row in positive_train).most_common(1)[0][0]
            constants = [{"target_present": True, "raw_bbox": mean_box, "raw_class_id": majority,
                          "latency_ms": 0.0} for _ in records]
            constant = calculate_metrics(records, constants)[0]
            constant["latency_ms_mean"] = constant["latency_ms_p50"] = None
            constant["description"] = "Always-present mean positive box and majority class computed only from the selected dataset's training split; ignores image and instruction."
            constant["bbox"] = mean_box
            constant["class_id"] = majority
            baseline_metrics["training_mean_box"] = constant
        else:
            baseline_metrics["training_mean_box"] = {"available": False, "reason": "No positive training examples"}
    return metrics, baseline_metrics, examples if include_examples else []


def evaluate(run_id: str, checkpoint: str, version_id: str, split: str = "test", threshold: float = 0.5) -> dict:
    if split not in {"train", "val", "test"}:
        raise ValueError("Evaluation split must be train, val, or test")
    threshold = _check_threshold(threshold)
    manifest = verify_version(version_id)
    records = [row for row in manifest["records"] if row["split"] == split]
    if not records:
        raise ValueError(f"The selected dataset has no {split} examples")
    with INFERENCE_LOCK:
        model, tokenizer, state, device, path = load_model(run_id, checkpoint)
        if manifest["classes"] != state["classes"]:
            raise ValueError("Evaluation dataset class mapping does not match the selected model")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics, baselines, examples = evaluate_records(model, tokenizer, state["config"], state["classes"],
                                                        records, device, threshold,
                                                        baseline_records=[r for r in manifest["records"] if r["split"] == "train"])
        return {"id": storage.uid(), "run_id": run_id, "checkpoint": path.name, "version_id": version_id,
                "split": split, "threshold": threshold, "metrics": metrics, "baselines": baselines,
                "examples": examples, "created_at": storage.now(),
                "synthetic": bool(records) and all(row.get("synthetic", False) for row in records)}

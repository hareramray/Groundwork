"""Checks for the file-based model adapter, including safe loading boundaries."""
import hashlib
import io
import sqlite3

import pytest
import torch
from PIL import Image

from grounding.agent_model import FileGrounder
from grounding.ml import ARCHITECTURE, PREPROCESSING, Grounder, Tokenizer, capture_rng, preprocess_image
from grounding.training import default_config


class _AdditionalCheckpointObject:
    pass


class _PickleCode:
    def __init__(self, destination):
        self.destination = destination

    def __reduce__(self):
        # Must never execute unless an application explicitly uses unsafe pickle.
        return eval, (f"__import__('pathlib').Path({str(self.destination)!r}).write_text('executed')",)


@pytest.fixture
def payload():
    torch.set_num_threads(2)
    config = {**default_config(), "image_size": 32, "width": 8, "text_dim": 16, "max_tokens": 8}
    tokenizer = Tokenizer.build(["click the button", "search box"], config["max_tokens"])
    model = Grounder(len(tokenizer.vocabulary), 2, config)
    # This constant fixture verifies integration mechanics, not learned accuracy.
    with torch.no_grad():
        model.box_head.weight.zero_()
        model.box_head.bias.zero_()
        model.presence_head.weight.zero_()
        model.presence_head.bias.fill_(2)
        model.class_head.weight.zero_()
        model.class_head.bias.copy_(torch.tensor([2.0, -2.0]))
    return {"schema": 1, "kind": "inference_export", "architecture": ARCHITECTURE,
            "preprocessing": PREPROCESSING, "config": config, "tokenizer": tokenizer.to_dict(),
            "classes": ["button", "textbox"], "model": model.state_dict(), "run_id": "test-only"}


@pytest.fixture
def screenshot():
    stream = io.BytesIO()
    Image.new("RGB", (213, 97), (48, 99, 157)).save(stream, format="PNG")
    return stream.getvalue()


def _save(tmp_path, payload):
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    return path


@pytest.mark.parametrize("kind", ["inference_export", "training_checkpoint"])
def test_loads_generated_formats_safely_once_without_database_and_preserves_source(tmp_path, monkeypatch, payload, screenshot, kind):
    payload["kind"] = kind
    if kind == "training_checkpoint":
        payload.update(rng=capture_rng(), optimizer={"state": {}, "param_groups": []}, scheduler={}, scaler={},
                       epoch=1, cursor=0, global_step=1, best_metric=0.0,
                       execution={"device": "cpu", "torch": torch.__version__, "mixed_precision": False})
    path = _save(tmp_path, payload)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    source_mtime = path.stat().st_mtime_ns
    monkeypatch.setattr(sqlite3, "connect", lambda *args, **kwargs: pytest.fail("A direct PT load must not use SQLite"))
    grounder = FileGrounder(path, device="cpu")
    model_identity = id(grounder.model)
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("A prediction must not reload its model"))
    first = grounder.predict(screenshot, "click the button")
    second = grounder.predict(screenshot, "click the button")
    assert first["bbox"] == second["bbox"]
    assert id(grounder.model) == model_identity
    assert grounder.metadata["kind"] == kind
    assert grounder.metadata["load_mode"] == "weights_only"
    assert grounder.metadata["checkpoint_sha256"] == source_hash
    assert not grounder.model.training
    assert path.stat().st_mtime_ns == source_mtime
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    assert "optimizer" not in vars(grounder)


def test_prediction_reuses_training_preprocessing_and_original_dimensions(tmp_path, payload, screenshot):
    grounder = FileGrounder(_save(tmp_path, payload), device="cpu")
    seen = []
    hook = grounder.model.register_forward_pre_hook(lambda model, args: seen.append(args))
    prediction = grounder.predict(screenshot, "CLICK the mysterious button")
    hook.remove()
    with Image.open(io.BytesIO(screenshot)) as original:
        expected = preprocess_image(original, 32)
    assert torch.equal(seen[0][0][0], expected)
    assert tuple(seen[0][0].shape) == (1, 3, 32, 32)
    assert prediction["width"] == 213
    assert prediction["height"] == 97
    assert prediction["class_name"] == "button"
    assert prediction["presence_score"] == pytest.approx(float(torch.tensor(2).sigmoid()))
    assert prediction["unknown_words"] == ["mysterious"]
    box = prediction["bbox"]
    assert prediction["bbox_pixels"] == pytest.approx([box[0] * 213, box[1] * 97, box[2] * 213, box[3] * 97])
    assert prediction["click_point_pixels"] == pytest.approx([prediction["click_point"][0] * 213,
                                                            prediction["click_point"][1] * 97])
    assert prediction["process_rss_mb"] > 0
    assert prediction["memory_mb"] == prediction["process_rss_mb"]
    assert prediction["gpu_peak_allocated_mb"] == 0
    assert prediction["latency_ms"] > 0
    assert "excludes model loading" in prediction["latency_scope"]
    assert "not a calibrated" in prediction["score_description"]


def test_threshold_abstention_and_override_never_provide_click_on_absence(tmp_path, payload, screenshot):
    grounder = FileGrounder(_save(tmp_path, payload), device="cpu", threshold=.99)
    result = grounder.predict(screenshot, "button")
    assert result["target_present"] is False
    for key in ("bbox", "bbox_pixels", "click_point", "click_point_pixels", "class_id", "class_name"):
        assert result[key] is None
    assert len(result["raw_bbox"]) == 4
    assert grounder.predict(screenshot, "button", threshold=.5)["target_present"] is True
    assert grounder.threshold == .99


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -.1, 1.1, True, ".5"])
def test_invalid_thresholds_fail_before_predicting(tmp_path, payload, screenshot, threshold):
    path = _save(tmp_path, payload)
    with pytest.raises(ValueError, match="threshold"):
        FileGrounder(path, threshold=threshold)
    grounder = FileGrounder(path, device="cpu")
    with pytest.raises(ValueError, match="threshold"):
        grounder.predict(screenshot, "button", threshold=threshold)


@pytest.mark.parametrize("field,value,match", [
    ("schema", 2, "format"), ("schema", True, "format"), ("kind", "state_dict", "format"),
    ("kind", [], "format"), ("config", {123: 1}, "config"),
    ("architecture", "other-model", "architecture"), ("preprocessing", "crop", "preprocessing"),
    ("classes", [], "classes"), ("classes", ["button", "button"], "classes"),
    ("classes", [123], "classes"), ("classes", [" "], "classes"), ("config", None, "config"),
    ("tokenizer", {"vocabulary": ["<pad>", "<unk>"], "max_tokens": 4}, "tokenizer"),
    ("tokenizer", {"vocabulary": ["bad", "<unk>"], "max_tokens": 8}, "vocabulary"),
    ("tokenizer", {"vocabulary": ["<pad>", "<unk>", []], "max_tokens": 8}, "vocabulary"),
    ("model", {"something": torch.ones(1)}, "model state"),
])
def test_rejects_unsupported_or_malformed_metadata(tmp_path, payload, field, value, match):
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        FileGrounder(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("field", ["model", "tokenizer", "classes", "config"])
def test_missing_metadata_is_explicit(tmp_path, payload, field):
    del payload[field]
    with pytest.raises(ValueError, match=f"missing {field}"):
        FileGrounder(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("field,value", [("image_size", 0), ("width", True), ("text_dim", 17),
                                        ("learning_rate", float("nan")), ("max_tokens", 0)])
def test_invalid_config_is_rejected(tmp_path, payload, field, value):
    payload["config"][field] = value
    with pytest.raises(ValueError, match=field):
        FileGrounder(_save(tmp_path, payload), device="cpu")


def test_missing_architecture_dimension_is_not_filled_from_defaults(tmp_path, payload):
    del payload["config"]["image_size"]
    with pytest.raises(ValueError, match="missing image_size"):
        FileGrounder(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("bad_tensor", [torch.zeros(3), torch.zeros(4, dtype=torch.int64),
                                      torch.tensor([0.0, float("nan"), 0.0, 0.0]),
                                      torch.tensor([0.0, float("inf"), 0.0, 0.0])])
def test_malformed_tensors_cannot_be_loaded(tmp_path, payload, bad_tensor):
    payload["model"]["box_head.bias"] = bad_tensor
    with pytest.raises(ValueError, match="Invalid model tensor box_head.bias"):
        FileGrounder(_save(tmp_path, payload), device="cpu")


@pytest.mark.parametrize("bad_state", [{"foo": torch.ones(1)}, torch.ones(1), b"not a torch archive"])
def test_rejects_arbitrary_state_or_file(tmp_path, bad_state):
    path = tmp_path / "arbitrary.pt"
    if isinstance(bad_state, bytes):
        path.write_bytes(bad_state)
    else:
        torch.save(bad_state, path)
    with pytest.raises(ValueError):
        FileGrounder(path, device="cpu")


def test_default_safe_loader_does_not_execute_arbitrary_pickle(tmp_path, payload):
    marker = tmp_path / "should-not-exist"
    payload["extra"] = _PickleCode(marker)
    with pytest.raises(ValueError, match="trust-checkpoint"):
        FileGrounder(_save(tmp_path, payload), device="cpu")
    assert not marker.exists()


def test_unsafe_pickle_fallback_requires_explicit_trust_and_records_it(tmp_path, payload):
    payload["extra"] = _AdditionalCheckpointObject()
    path = _save(tmp_path, payload)
    with pytest.raises(ValueError, match="trust-checkpoint"):
        FileGrounder(path, device="cpu")
    grounder = FileGrounder(path, device="cpu", trust_checkpoint=True)
    assert grounder.metadata["load_mode"] == "trusted_pickle"


def test_cpu_override_and_auto_fallback_do_not_depend_on_training_device(tmp_path, monkeypatch, payload):
    payload["config"]["device"] = "cuda"
    path = _save(tmp_path, payload)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert FileGrounder(path, device="auto").metadata["device"] == "cpu"
    assert FileGrounder(path, device="cpu").metadata["device"] == "cpu"
    with pytest.raises(ValueError, match="CUDA was requested"):
        FileGrounder(path, device="cuda")
    with pytest.raises(ValueError, match="Device must be"):
        FileGrounder(path, device="remote")


def test_metadata_is_a_snapshot(tmp_path, payload):
    grounder = FileGrounder(_save(tmp_path, payload), device="cpu")
    metadata = grounder.metadata
    metadata["classes"][0] = "changed"
    metadata["config"]["image_size"] = 4096
    assert grounder.classes[0] == "button"
    assert grounder.config["image_size"] == 32


@pytest.mark.parametrize("part,value", [("presence_logits", torch.tensor([float("nan")])),
                                       ("class_logits", torch.tensor([[float("inf"), 0.0]])),
                                       ("bbox", torch.tensor([[.9, .2, .1, .5]])),
                                       ("bbox", torch.tensor([[.1, .2, 1.1, .5]]))])
def test_invalid_runtime_predictions_never_return_target(tmp_path, monkeypatch, payload, screenshot, part, value):
    grounder = FileGrounder(_save(tmp_path, payload), device="cpu")
    invalid = {"bbox": torch.tensor([[.1, .2, .4, .5]]), "class_logits": torch.zeros(1, 2),
               "presence_logits": torch.ones(1)}
    invalid[part] = value
    monkeypatch.setattr(grounder.model, "forward", lambda *args: invalid)
    with pytest.raises(ValueError, match="invalid|non-finite"):
        grounder.predict(screenshot, "button")


def test_rejects_invalid_image_and_instruction(tmp_path, payload, screenshot):
    grounder = FileGrounder(_save(tmp_path, payload), device="cpu")
    with pytest.raises(ValueError, match="instruction"):
        grounder.predict(screenshot, " ")
    with pytest.raises(ValueError, match="PNG"):
        grounder.predict(b"bad image", "button")
    stream = io.BytesIO()
    Image.new("RGB", (20, 20)).save(stream, format="JPEG")
    with pytest.raises(ValueError, match="PNG"):
        grounder.predict(stream.getvalue(), "button")

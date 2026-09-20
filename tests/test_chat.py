"""Teaching chat to a saved grounding model while preserving its grounding ability."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient

from grounding import chat, dataset, storage, training
from grounding.agent_model import FileGrounder
from grounding.api import app
from grounding.ml import Grounder, Tokenizer, make_batch
from grounding.synthetic import generate


@pytest.fixture
def local_data(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setenv("GROUNDING_DATA_DIR", str(tmp_path))
    storage.init_db()
    return tmp_path


@pytest.fixture
def client(local_data):
    with TestClient(app) as instance:
        yield instance


@pytest.fixture
def grounding_parent(local_data):
    generate(count=6, seed=6, reviewed=True)
    version = dataset.create_version({"name": "Grounding source", "seed": 4,
                                      "train_ratio": .67, "val_ratio": .17})
    config = {**training.default_config(), "image_size": 32, "batch_size": 3,
              "epochs": 1, "width": 8, "text_dim": 64, "max_tokens": 16,
              "device": "cpu", "checkpoint_every": 1}
    run = training.create_run({"name": "My trained grounding model", "version_id": version["id"],
                               "config": config})
    assert training.run_worker(run["id"])["status"] == "completed"
    path = training.resolve_checkpoint(run["id"])
    return {"run": storage.get("run", run["id"]), "path": path,
            "checkpoint": training.read_checkpoint(path), "version": version}


def create_chat(parent, **payload):
    return chat.create_run({"source_run_id": parent["run"]["id"], **payload})


def greetings():
    return [chat.save_example({"prompt": prompt, "response": response})
            for prompt, response in (("hey", "Hello!"), ("bye", "Goodbye!"))]


def configuration(**changes):
    return {**chat.default_config(), "device": "cpu", **changes}


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


@pytest.mark.parametrize("changes", [
    {"epochs": 0}, {"epochs": True}, {"epochs": 1.5},
    {"batch_size": 0}, {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"learning_rate": float("inf")}, {"seed": -1}, {"device": "invalid"},
    {"unknown_setting": 1},
])
def test_chat_configuration_rejects_invalid_values(changes):
    with pytest.raises(ValueError):
        chat.validate_config(changes)


@pytest.mark.parametrize("field,value", [
    ("prompt", ""), ("prompt", " \n\t "), ("prompt", "x" * 501),
    ("response", ""), ("response", " \n\t "), ("response", "x" * 501),
    ("prompt", None), ("response", 123),
])
def test_invalid_chat_edit_preserves_existing_example(local_data, field, value):
    saved = chat.save_example({"prompt": "hey", "response": "Hello!"})
    with pytest.raises(ValueError):
        chat.save_example({"prompt": "hey", "response": "Hello!", field: value}, saved["id"])
    assert storage.get("chat_example", saved["id"]) == saved


def test_empty_dataset_and_unknown_examples_fail_without_writes(grounding_parent):
    with pytest.raises(ValueError):
        create_chat(grounding_parent, config=configuration(epochs=1))
    with pytest.raises(KeyError):
        chat.save_example({"prompt": "hey", "response": "Hello!"}, "missing")
    with pytest.raises(KeyError):
        chat.delete_example("missing")
    assert storage.list_items("chat_example") == []
    assert storage.list_items("chat_run") == []


def test_chat_api_examples_defaults_validation_and_storage_isolation(client):
    assert client.get("/api/chat/examples").json() == []
    defaults = client.get("/api/chat/training/defaults")
    assert defaults.status_code == 200
    assert defaults.json() == chat.default_config()
    created = client.post("/api/chat/examples", json={"prompt": "hey", "response": "Hello!"})
    assert created.status_code == 200, created.text
    example = created.json()
    assert client.get("/api/chat/examples").json() == [example]
    changed = client.put(f"/api/chat/examples/{example['id']}", json={"prompt": "hey", "response": "Hi there!"})
    assert changed.status_code == 200 and changed.json()["response"] == "Hi there!"
    rejected = client.put(f"/api/chat/examples/{example['id']}", json={"prompt": "", "response": "bad"})
    assert rejected.status_code == 422
    assert client.get("/api/chat/examples").json() == [changed.json()]
    assert client.post("/api/chat/examples", json={"prompt": "hey", "response": "x" * 501}).status_code == 422
    assert client.put("/api/chat/examples/missing", json={"prompt": "hey", "response": "Hello!"}).status_code == 404
    assert client.delete(f"/api/chat/examples/{example['id']}").status_code == 200
    assert client.get("/api/chat/examples").json() == []
    assert client.delete(f"/api/chat/examples/{example['id']}").status_code == 404
    assert storage.list_items("image") == []
    assert storage.list_items("run") == []
    assert storage.list_items("version") == []


def test_chat_api_run_creation_launch_failure_and_missing_resources(client, grounding_parent, monkeypatch):
    greetings()
    launched = []

    missing_source = client.post("/api/chat/runs", json={"name": "Needs my grounding model"})
    assert missing_source.status_code == 422
    assert storage.list_items("chat_run") == []
    sources = client.get("/api/chat/sources")
    assert sources.status_code == 200
    assert any(source["id"] == grounding_parent["run"]["id"] and source["kind"] == "grounding"
               for source in sources.json())

    def launch(run_id):
        launched.append(run_id)
        return storage.get("chat_run", run_id)

    monkeypatch.setattr(chat, "launch_run", launch)
    response = client.post("/api/chat/runs", json={"source_run_id": grounding_parent["run"]["id"], "name": "My greetings", "config": configuration(epochs=2)})
    assert response.status_code == 200, response.text
    run = response.json()
    assert launched == [run["id"]]
    assert client.get(f"/api/chat/runs/{run['id']}").json()["name"] == "My greetings"
    assert [item["id"] for item in client.get("/api/chat/runs").json()] == [run["id"]]
    assert client.post("/api/chat/runs", json={"config": {"epochs": 0}}).status_code == 422
    assert launched == [run["id"]]
    assert client.get("/api/chat/runs/missing").status_code == 404
    assert client.post("/api/chat/predict", json={"run_id": "missing", "message": "hey"}).status_code == 404
    assert client.post("/api/chat/predict", json={"run_id": run["id"], "message": " "}).status_code == 422
    assert client.get(f"/api/chat/runs/{run['id']}/download").status_code in (404, 422)

    def failed_launch(run_id):
        raise OSError("Simulated worker launch failure")

    monkeypatch.setattr(chat, "launch_run", failed_launch)
    failed = client.post("/api/chat/runs", json={"source_run_id": grounding_parent["run"]["id"], "name": "Launch failure", "config": configuration(epochs=1)})
    assert failed.status_code == 422
    persisted = next(item for item in storage.list_items("chat_run") if item["name"] == "Launch failure")
    assert persisted["status"] == "error"
    assert "Simulated worker launch failure" in persisted["error"]


def test_chat_api_stop_resume_routes(client, monkeypatch):
    calls = []

    def control(run_id, action):
        calls.append((run_id, action))
        return {"id": run_id, "status": "stopped" if action == "stop" else "running"}

    monkeypatch.setattr(chat, "control_run", control)
    assert client.post("/api/chat/runs/chat-test/stop").json()["status"] == "stopped"
    assert client.post("/api/chat/runs/chat-test/resume").json()["status"] == "running"
    assert calls == [("chat-test", "stop"), ("chat-test", "resume")]
    assert client.post("/api/chat/runs/chat-test/unknown").status_code == 404
    assert len(calls) == 2


def test_same_model_learns_chat_preserves_grounding_and_exports_both(client, grounding_parent, local_data):
    examples = greetings()
    parent_hash = hashlib.sha256(grounding_parent["path"].read_bytes()).hexdigest()
    parent_metadata = copy.deepcopy(grounding_parent["run"])
    parent_checkpoint = grounding_parent["checkpoint"]
    run = create_chat(grounding_parent, name="Learn my replies", config=configuration(epochs=200))
    original_run = copy.deepcopy(run)
    # Later edits and deletion must not silently rewrite an experiment's examples.
    chat.save_example({"prompt": "hey", "response": "This is a later edit."}, examples[0]["id"])
    chat.delete_example(examples[1]["id"])
    assert storage.get("chat_run", run["id"]) == original_run

    first_epoch = chat.run_worker(run["id"], stop_after_epochs=1)
    assert first_epoch["status"] == "stopped"
    initial_loss = first_epoch["progress"]["loss"]
    finished = chat.run_worker(run["id"])
    assert finished["status"] == "completed", finished
    assert finished["progress"]["epoch"] == 200
    assert finished["progress"]["global_step"] == 200
    assert finished["progress"]["loss"] < initial_loss * 0.1
    checkpoint = torch.load(chat.resolve_checkpoint(run["id"]), map_location="cpu", weights_only=True)
    assert checkpoint["kind"] == "grounding_chat_training_checkpoint"
    assert checkpoint["architecture"] == "cnn-gru-grounding-chat-v2"
    assert checkpoint["config"] == parent_checkpoint["config"]
    assert checkpoint["tokenizer"] == parent_checkpoint["tokenizer"]
    assert checkpoint["classes"] == parent_checkpoint["classes"]
    for name, weight in parent_checkpoint["model"].items():
        assert torch.equal(checkpoint["model"][name], weight), f"Grounding parameter changed: {name}"

    # Both tasks use the very same text encoder, with the original grounding weights frozen.
    tokenizer = Tokenizer.from_dict(checkpoint["tokenizer"])
    before = Grounder(len(tokenizer.vocabulary), len(checkpoint["classes"]), checkpoint["config"]).eval()
    before.load_state_dict(parent_checkpoint["model"])
    after = Grounder(len(tokenizer.vocabulary), len(checkpoint["classes"]), checkpoint["config"],
                     len(checkpoint["chat_tokenizer"]["characters"]) + 4).eval()
    after.load_state_dict(checkpoint["model"])
    batch = make_batch(grounding_parent["version"]["records"][:2], tokenizer, checkpoint["config"],
                       local_data, torch.device("cpu"))
    encoder_calls = []
    handle = after.text_encoder.register_forward_hook(lambda *args: encoder_calls.append(True))
    with torch.inference_mode():
        after.encode_chat(torch.tensor([[4, 5, 2]]), torch.tensor([3]))
        assert encoder_calls == [True]
        original_output = before(batch["images"], batch["tokens"], batch["lengths"])
        unified_output = after(batch["images"], batch["tokens"], batch["lengths"])
    handle.remove()
    assert_nested_equal(original_output, unified_output)
    for example in examples:
        prediction = client.post("/api/chat/predict", json={"run_id": run["id"], "message": example["prompt"]})
        assert prediction.status_code == 200, prediction.text
        assert prediction.json()["reply"] == example["response"]
        assert prediction.json()["run_id"] == run["id"]
        assert prediction.json()["checkpoint"]
    assert chat.predict(run["id"], "hey\N{WAVING HAND SIGN}")["unknown_characters"] == 1

    downloaded = client.get(f"/api/chat/runs/{run['id']}/download")
    assert downloaded.status_code == 200, downloaded.text
    assert "attachment" in downloaded.headers["content-disposition"]
    exported = torch.load(io.BytesIO(downloaded.content), map_location="cpu", weights_only=True)
    assert exported["kind"] == "inference_export"
    assert exported["tokenizer"] == checkpoint["tokenizer"]
    assert exported["chat_tokenizer"] == checkpoint["chat_tokenizer"]
    assert_nested_equal(exported["model"], checkpoint["model"])
    assert "optimizer" not in exported and "examples" not in exported
    export_path = chat.export_path(run["id"])
    assert training.read_checkpoint(export_path)["kind"] == "inference_export"
    record = grounding_parent["version"]["records"][0]
    screenshot = (local_data / record["image_path"]).read_bytes()
    original_prediction = FileGrounder(grounding_parent["path"], device="cpu").predict(screenshot, record["instruction"])
    loaded_model = FileGrounder(export_path, device="cpu")
    unified_prediction = loaded_model.predict(screenshot, record["instruction"])
    for example in examples:
        assert loaded_model.chat(example["prompt"])["reply"] == example["response"]
    for key in ("raw_bbox", "presence_score", "raw_class_id", "target_present", "bbox", "click_point"):
        assert unified_prediction[key] == original_prediction[key]
    api_grounding = client.post("/api/predict", files={"file": ("screen.png", screenshot, "image/png")},
                               data={"instruction": record["instruction"], "run_id": run["id"], "checkpoint": "latest"})
    assert api_grounding.status_code == 200, api_grounding.text
    assert api_grounding.json()["raw_bbox"] == original_prediction["raw_bbox"]
    assert api_grounding.json()["presence_score"] == original_prediction["presence_score"]
    assert hashlib.sha256(grounding_parent["path"].read_bytes()).hexdigest() == parent_hash
    assert storage.get("run", grounding_parent["run"]["id"]) == parent_metadata
    assert any(item["run_id"] == run["id"] for item in storage.list_items("export"))

    # Continuing screenshot training from this same export must retain learned replies too.
    exported_record = chat.export_model(run["id"])
    retrained = training.create_run({"mode": "retrain", "version_id": grounding_parent["version"]["id"],
                                     "source_export_id": exported_record["id"], "config": parent_checkpoint["config"]})
    assert training.run_worker(retrained["id"])["status"] == "completed"
    retrained_state = training.read_checkpoint(training.resolve_checkpoint(retrained["id"]))
    for name, weight in checkpoint["model"].items():
        if name.startswith(("chat_", "text_encoder.")):
            assert torch.equal(retrained_state["model"][name], weight), name
    continued_export = training.export_model(retrained["id"])
    continued_model = FileGrounder(storage.safe_path(continued_export["path"]), device="cpu")
    for example in examples:
        assert continued_model.chat(example["prompt"])["reply"] == example["response"]
    assert continued_model.predict(screenshot, record["instruction"])["width"] == record["width"]


def test_chat_resume_restores_optimizer_and_remaining_epochs(grounding_parent):
    greetings()
    config = configuration(epochs=6, batch_size=1)
    uninterrupted = create_chat(grounding_parent, config=config)
    resumed = create_chat(grounding_parent, config=config)
    assert chat.run_worker(uninterrupted["id"])["status"] == "completed"
    stopped = chat.run_worker(resumed["id"], stop_after_epochs=2)
    assert stopped["status"] == "stopped"
    assert stopped["progress"]["epoch"] == 2
    assert stopped["progress"]["global_step"] == 4
    intermediate = torch.load(chat.resolve_checkpoint(resumed["id"]), map_location="cpu", weights_only=True)
    assert intermediate["epoch"] == 2 and intermediate["cursor"] == 0
    assert intermediate["optimizer"]["state"]

    final_run = chat.run_worker(resumed["id"])
    assert final_run["status"] == "completed"
    assert final_run["progress"]["epoch"] == 6
    assert final_run["progress"]["global_step"] == 12
    full = torch.load(chat.resolve_checkpoint(uninterrupted["id"]), map_location="cpu", weights_only=True)
    final = torch.load(chat.resolve_checkpoint(resumed["id"]), map_location="cpu", weights_only=True)
    for key in ("model", "optimizer", "epoch", "cursor", "global_step", "loss"):
        assert_nested_equal(full[key], final[key])


def test_chat_can_continue_same_model_with_new_characters(grounding_parent, local_data):
    greetings()
    parent = create_chat(grounding_parent, config=configuration(epochs=2))
    assert chat.run_worker(parent["id"])["status"] == "completed"
    parent_path = chat.resolve_checkpoint(parent["id"])
    parent_hash = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    parent_state = torch.load(parent_path, map_location="cpu", weights_only=True)
    old_characters = parent_state["chat_tokenizer"]["characters"]
    chat.save_example({"prompt": "yo?", "response": "Hi \N{SPARKLES}"})
    child = chat.create_run({"source_chat_run_id": parent["id"], "config": configuration(epochs=2)})
    assert child["source_chat_run_id"] == parent["id"]
    initial = torch.load(local_data / "chat" / "runs" / child["id"] / "initial_weights.pt",
                         map_location="cpu", weights_only=True)
    assert initial["chat_tokenizer"]["characters"][:len(old_characters)] == old_characters
    assert "\N{SPARKLES}" in initial["chat_tokenizer"]["characters"]
    for name, original in parent_state["model"].items():
        expanded = initial["model"][name]
        if name in {"chat_embedding.weight", "chat_output.weight", "chat_output.bias"}:
            expanded = expanded[:original.shape[0]]
        assert torch.equal(expanded, original), f"Parent weight was not retained: {name}"
    assert chat.run_worker(child["id"])["status"] == "completed"
    child_state = torch.load(chat.resolve_checkpoint(child["id"]), map_location="cpu", weights_only=True)
    for name, original in grounding_parent["checkpoint"]["model"].items():
        assert torch.equal(child_state["model"][name], original)
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == parent_hash


def test_chat_rejects_untrained_and_changed_experiments(local_data, grounding_parent):
    greetings()
    run = create_chat(grounding_parent, config=configuration(epochs=2))
    with pytest.raises(ValueError, match="Train"):
        chat.resolve_checkpoint(run["id"])
    with pytest.raises(ValueError):
        chat.predict(run["id"], "hey")
    with pytest.raises(ValueError):
        chat.export_model(run["id"])
    path = local_data / "chat" / "runs" / run["id"] / "experiment.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    snapshot["examples"][0]["response"] = "Changed behind the training run"
    path.write_text(json.dumps(snapshot), encoding="utf-8")
    with pytest.raises(ValueError, match="immutable experiment changed"):
        chat.run_worker(run["id"])
    assert storage.get("chat_run", run["id"])["status"] == "error"


def test_chat_and_grounding_cannot_launch_over_each_other(grounding_parent, monkeypatch):
    from grounding import training

    greetings()
    run = create_chat(grounding_parent, config=configuration(epochs=1))
    storage.put("run", "grounding-worker", {"id": "grounding-worker", "status": "running"})
    monkeypatch.setattr(training, "_worker_alive", lambda record: record["status"] == "running")
    monkeypatch.setattr(chat, "_worker_alive", lambda record: record["status"] == "running")

    def unexpected_worker(*args, **kwargs):
        pytest.fail("A second training process must not be started")

    monkeypatch.setattr(training.subprocess, "Popen", unexpected_worker)
    with pytest.raises(ValueError, match="active"):
        chat.launch_run(run["id"])
    storage.patch("run", "grounding-worker", {"status": "stopped"})
    storage.patch("chat_run", run["id"], {"status": "running"})
    with pytest.raises(ValueError, match="active"):
        training.launch_run("grounding-worker")


def test_chat_recovery_rejects_unrelated_process_reusing_worker_pid(grounding_parent):
    greetings()
    run = create_chat(grounding_parent, config=configuration(epochs=1))
    # The test process exists, but it is not this run's chat worker.
    stale = storage.patch("chat_run", run["id"], {"status": "running", "pid": os.getpid()})
    assert not chat._worker_alive(stale)
    chat.recover_runs()
    recovered = storage.get("chat_run", run["id"])
    assert recovered["status"] == "interrupted"
    assert recovered["pid"] is None


def test_chat_stop_before_learning_disables_prediction_and_resumes(client, local_data, grounding_parent, monkeypatch):
    greetings()
    run = create_chat(grounding_parent, config=configuration(epochs=2))
    storage.patch("chat_run", run["id"], {"status": "queued"})
    assert chat.control_run(run["id"], "stop")["requested_control"] == "stop"
    stopped = chat.run_worker(run["id"])
    directory = local_data / "chat" / "runs" / run["id"]
    assert stopped["status"] == "stopped"
    assert stopped["progress"]["global_step"] == 0
    assert stopped["latest_checkpoint"] is None
    assert (directory / "checkpoints" / "latest.pt").is_file()
    assert client.post("/api/chat/predict", json={"run_id": run["id"], "message": "hey"}).status_code == 422
    assert client.get(f"/api/chat/runs/{run['id']}/download").status_code == 422

    launched = []

    def launch(arguments, **kwargs):
        launched.append(arguments)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(chat.subprocess, "Popen", launch)
    monkeypatch.setattr(chat, "PROCESSES", {})
    assert chat.control_run(run["id"], "resume")["status"] == "queued"
    assert launched[0][-2:] == ["--run", run["id"]]
    assert not (directory / "control.json").exists()
    completed = chat.run_worker(run["id"])
    assert completed["status"] == "completed"
    assert completed["progress"]["global_step"] == 2
    assert completed["latest_checkpoint"] == "latest.pt"
    assert chat.resolve_checkpoint(run["id"]).is_file()

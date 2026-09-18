"""Data and API acceptance checks. Every test uses isolated local storage."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile

from PIL import Image
import pytest
from fastapi.testclient import TestClient

from grounding import dataset, storage
from grounding.api import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "ROOT", tmp_path)
    monkeypatch.setenv("GROUNDING_DATA_DIR", str(tmp_path))
    storage.init_db()
    with TestClient(app) as instance:
        yield instance


def upload(client, color="navy"):
    output = io.BytesIO()
    Image.new("RGB", (320, 160), color=color).save(output, "PNG")
    response = client.post("/api/images", files=[("files", ("screenshot.png", output.getvalue(), "image/png"))])
    assert response.status_code == 200, response.text
    return response.json()[0]


def annotation(prefix="one", status="draft"):
    return {
        "group": "store-template-a",
        "elements": [{"id": f"{prefix}-element", "class_id": 0, "label": "Submit", "bbox": [0.1, 0.2, 0.8, 0.6], "click_point": [0.3, 0.4]}],
        "examples": [
            {"id": f"{prefix}-positive", "instruction": "Click the submit button", "target_present": True,
             "element_id": f"{prefix}-element", "status": status, "ambiguous": False},
            {"id": f"{prefix}-absent", "instruction": "Find a search textbox", "target_present": False,
             "element_id": None, "status": status, "ambiguous": False},
        ],
    }


def create_version(client, name="Test version", **kwargs):
    response = client.post("/api/versions", json={"name": name, "seed": 17, "group_by": "group", **kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def test_upload_annotation_edit_and_explicit_review(client):
    image = upload(client)
    assert image["width"] == 320 and image["height"] == 160
    assert image["status"] == "unannotated"
    payload = annotation()
    saved = client.put(f"/api/images/{image['id']}", json=payload)
    assert saved.status_code == 200
    assert saved.json()["status"] == "draft"
    assert client.get(f"/api/images/{image['id']}").json()["elements"] == payload["elements"]
    assert client.post("/api/versions", json={"name": "Must not train drafts"}).status_code == 422
    # Moving/resizing an element updates all its instruction references.
    payload["elements"][0].update(bbox=[0.2, 0.3, 0.9, 0.8], click_point=[0.5, 0.5])
    for example in payload["examples"]:
        example["status"] = "reviewed"
    saved = client.put(f"/api/images/{image['id']}", json=payload)
    assert saved.status_code == 200 and saved.json()["status"] == "reviewed"
    version = create_version(client)
    positive = next(r for r in version["records"] if r["target_present"])
    absent = next(r for r in version["records"] if not r["target_present"])
    assert positive["bbox"] == [0.2, 0.3, 0.9, 0.8]
    assert positive["click_point"] == [0.5, 0.5]
    assert absent["bbox"] is None and absent["class_id"] is None and absent["click_point"] is None
    assert positive["split"] == absent["split"]


@pytest.mark.parametrize("mutation,expected", [
    (lambda p: p["elements"][0].update(bbox=[0.8, 0.2, 0.1, 0.6]), "bounding box"),
    (lambda p: p["elements"][0].update(bbox=[-0.1, 0.2, 0.8, 0.6]), "bounding box"),
    (lambda p: p["elements"][0].update(click_point=[0.99, 0.99]), "click point"),
    (lambda p: p["elements"][0].update(class_id=99), "unknown class"),
    (lambda p: p["examples"][0].update(instruction="   "), "instruction"),
    (lambda p: p["examples"][0].update(element_id="missing"), "matching element"),
    (lambda p: p["examples"][1].update(element_id="one-element"), "null targets"),
    (lambda p: p["examples"][0].update(status="reviewed", ambiguous=True), "ambiguity"),
])
def test_annotation_errors_do_not_replace_saved_data(client, mutation, expected):
    image = upload(client)
    valid = annotation()
    assert client.put(f"/api/images/{image['id']}", json=valid).status_code == 200
    invalid = copy.deepcopy(valid)
    mutation(invalid)
    rejected = client.put(f"/api/images/{image['id']}", json=invalid)
    assert rejected.status_code == 422 and expected in rejected.text
    persisted = client.get(f"/api/images/{image['id']}").json()
    assert persisted["elements"] == valid["elements"] and persisted["examples"] == valid["examples"]


def test_deleted_target_cannot_leave_dangling_instruction(client):
    image = upload(client)
    payload = annotation()
    assert client.put(f"/api/images/{image['id']}", json=payload).status_code == 200
    payload["elements"] = []
    assert client.put(f"/api/images/{image['id']}", json=payload).status_code == 422
    payload["examples"] = [payload["examples"][1]]
    assert client.put(f"/api/images/{image['id']}", json=payload).status_code == 200


def test_synthetic_defaults_draft_and_excluded_not_versioned(client):
    response = client.post("/api/synthetic", json={"count": 6, "seed": 11})
    assert response.status_code == 200
    images = client.get("/api/images").json()
    assert len(images) == 6 and all(im["synthetic"] for im in images)
    assert all(im["status"] == "draft" for im in images)
    report = client.post("/api/datasets/validate", json={}).json()
    assert report["errors"] and report["stats"]["unreviewed"] == response.json()["examples"]
    image = images[0]
    payload = {"group": image["group"], "elements": image["elements"], "examples": image["examples"]}
    payload["examples"][0]["status"] = "reviewed"
    payload["examples"][1]["status"] = "excluded"
    payload["examples"][2]["ambiguous"] = True
    assert client.put(f"/api/images/{image['id']}", json=payload).status_code == 200
    version = create_version(client)
    assert len(version["records"]) == 1
    assert version["records"][0]["id"] == payload["examples"][0]["id"]


def test_synthetic_seed_reproduces_pixels_targets_and_instructions(client, tmp_path, monkeypatch):
    def generated_signature():
        response = client.post("/api/synthetic", json={"count": 6, "seed": 731})
        assert response.status_code == 200
        signatures = []
        for im in sorted(client.get("/api/images").json(), key=lambda image: image["filename"]):
            positions = {element["id"]: index for index, element in enumerate(im["elements"])}
            signatures.append({
                "filename": im["filename"], "size": [im["width"], im["height"]], "group": im["group"],
                "pixels": hashlib.sha256((storage.ROOT / im["image_path"]).read_bytes()).hexdigest(),
                "elements": [{k: v for k, v in element.items() if k != "id"} for element in im["elements"]],
                "examples": [{**{k: v for k, v in example.items() if k not in {"id", "element_id"}},
                              "target_index": positions.get(example["element_id"])} for example in im["examples"]],
            })
        return signatures
    first = generated_signature()
    monkeypatch.setattr(storage, "ROOT", tmp_path / "same-seed-recreated")
    storage.init_db()
    assert generated_signature() == first


def test_group_and_duplicate_screenshot_splits_are_leakage_safe(client):
    assert client.post("/api/synthetic", json={"count": 18, "seed": 91, "reviewed": True}).status_code == 200
    first = create_version(client)
    second = create_version(client, "Same seed")
    by_group, by_image = {}, {}
    for record in first["records"]:
        by_group.setdefault(record["group"], set()).add(record["split"])
        by_image.setdefault(record["image_id"], set()).add(record["split"])
    assert all(len(splits) == 1 for splits in by_group.values())
    assert all(len(splits) == 1 for splits in by_image.values())
    assert {r["split"] for r in first["records"]} == {"train", "val", "test"}
    assert {r["id"]: r["split"] for r in first["records"]} == {r["id"]: r["split"] for r in second["records"]}
    # Even separately uploaded identical pixels with different group labels cannot leak.
    duplicates = []
    for number in range(2):
        im = upload(client, "orange")
        payload = annotation(f"duplicate-{number}", "reviewed")
        payload["group"] = f"different-group-{number}"
        assert client.put(f"/api/images/{im['id']}", json=payload).status_code == 200
        duplicates.append(im["id"])
    deduplicated = create_version(client, "Duplicate pixels")
    assert len({r["split"] for r in deduplicated["records"] if r["image_id"] in duplicates}) == 1


def test_snapshot_survives_live_edits_deletion_and_detects_tampering(client):
    im = upload(client)
    payload = annotation(status="reviewed")
    assert client.put(f"/api/images/{im['id']}", json=payload).status_code == 200
    version = create_version(client)
    source = storage.ROOT / version["records"][0]["image_path"]
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    payload["examples"][0]["instruction"] = "Updated description"
    assert client.put(f"/api/images/{im['id']}", json=payload).status_code == 200
    assert dataset.verify_version(version["id"])["records"] == version["records"]
    assert client.delete(f"/api/images/{im['id']}").status_code == 200
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    assert dataset.verify_version(version["id"])["fingerprint"] == version["fingerprint"]
    source.write_bytes(b"changed outside application")
    with pytest.raises(ValueError, match="image changed"):
        dataset.verify_version(version["id"])
    assert client.get(f"/api/versions/{version['id']}/export").status_code == 422


def test_snapshot_metadata_fingerprint_detects_changes(client):
    client.post("/api/synthetic", json={"count": 3, "seed": 19, "reviewed": True})
    version = create_version(client)
    path = storage.ROOT / "versions" / version["id"] / "manifest.json"
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["records"][0]["instruction"] = "Silently changed instruction"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata has changed"):
        dataset.verify_version(version["id"])


def test_export_import_preserves_examples_pixels_nulls_and_ids(client, tmp_path, monkeypatch):
    client.post("/api/synthetic", json={"count": 6, "seed": 47, "reviewed": True})
    version = create_version(client)
    response = client.get(f"/api/versions/{version['id']}/export")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert {"manifest.json", "records.jsonl"} <= set(archive.namelist())
        records = [json.loads(line) for line in archive.read("records.jsonl").decode().splitlines()]
        assert all(r["image_path"].startswith("images/") for r in records)
        hashes = {r["image_id"]: hashlib.sha256(archive.read(r["image_path"])).hexdigest() for r in records}
    # Importing into a separate repository must preserve stable example IDs and target data.
    monkeypatch.setattr(storage, "ROOT", tmp_path / "imported")
    storage.init_db()
    imported = client.post("/api/datasets/import", files={"file": ("dataset.zip", response.content, "application/zip")})
    assert imported.status_code == 200, imported.text
    assert imported.json()["examples"] == len(records) and imported.json()["images"] == 6
    actual = {r["id"]: r for r in dataset.records_from_images()}
    for expected in records:
        saved = actual[expected["id"]]
        for key in ("image_id", "width", "height", "instruction", "target_present", "class_id", "bbox", "click_point", "status", "group"):
            assert saved[key] == expected[key]
        assert hashlib.sha256((storage.ROOT / saved["image_path"]).read_bytes()).hexdigest() == hashes[saved["image_id"]]
    repeated = client.post("/api/datasets/import", files={"file": ("dataset.zip", response.content, "application/zip")})
    assert repeated.status_code == 422 and "Duplicate" in repeated.text
    assert len(dataset.records_from_images()) == len(records)


def make_bundle(records, raw_image: bytes, extra=None):
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"schema_version": 1, "classes": dataset.DEFAULT_CLASSES}))
        archive.writestr("records.jsonl", "".join(json.dumps(r) + "\n" for r in records))
        archive.writestr("images/sample.png", raw_image)
        if extra:
            archive.writestr(extra, b"not permitted")
    return content.getvalue()


@pytest.mark.parametrize("invalid_fields", [{"width": 999}, {"instruction": "x" * 2001}, {"group": "g" * 501}])
def test_invalid_import_is_validated_before_any_commit(client, invalid_fields):
    raw = io.BytesIO()
    Image.new("RGB", (20, 10)).save(raw, "PNG")
    record = {"id": "first", "image_id": "sample", "image_path": "images/sample.png", "width": 20, "height": 10,
              "instruction": "Find a button", "target_present": False, "class_id": None, "bbox": None,
              "click_point": None, "status": "draft", "ambiguous": False, "group": "a", "dataset_version": None}
    invalid = dict(record, id="second", **invalid_fields)
    bundle = make_bundle([record, invalid], raw.getvalue())
    response = client.post("/api/datasets/import", files={"file": ("invalid.zip", bundle, "application/zip")})
    assert response.status_code == 422
    assert client.get("/api/images").json() == []
    traversal = make_bundle([record], raw.getvalue(), "../outside.txt")
    response = client.post("/api/datasets/import", files={"file": ("unsafe.zip", traversal, "application/zip")})
    assert response.status_code == 422 and "Unsafe ZIP path" in response.text
    assert not (storage.ROOT.parent / "outside.txt").exists()


def test_class_configuration_protects_existing_annotation_meanings(client):
    names = ["button", "textbox", "link", "checkbox", "dropdown", "icon", "menu"]
    assert client.put("/api/classes", json={"classes": names}).status_code == 200
    im = upload(client)
    assert client.put(f"/api/images/{im['id']}", json=annotation()).status_code == 200
    assert client.put("/api/classes", json={"classes": ["textbox", "button"]}).status_code == 422
    assert client.put("/api/classes", json={"classes": ["button", "button"]}).status_code == 422
    assert client.get("/api/classes").json()["classes"] == names


def test_correction_is_always_draft_and_needs_review(client):
    im = upload(client)
    response = client.post("/api/predictions/correct", json={"image_id": im["id"], "instruction": "Find the missing checkbox", "target_present": False})
    assert response.status_code == 200
    assert response.json()["status"] == "draft"
    assert response.json()["examples"][0]["status"] == "draft"
    assert client.post("/api/versions", json={"name": "Unreviewed correction"}).status_code == 422


def test_metadata_persists_across_database_reinitialization(client):
    im = upload(client)
    payload = annotation(status="reviewed")
    client.put(f"/api/images/{im['id']}", json=payload)
    version = create_version(client)
    storage.init_db()
    assert client.get(f"/api/images/{im['id']}").json()["examples"] == payload["examples"]
    assert client.get(f"/api/versions/{version['id']}").json()["fingerprint"] == version["fingerprint"]


def test_local_api_rejects_remote_web_origins(client):
    response = client.post("/api/synthetic", headers={"Origin": "https://remote.example"}, json={"count": 3})
    assert response.status_code == 403
    assert client.get("/api/images").json() == []


def test_configurable_loopback_port_can_save_annotations(client):
    response = client.post("/api/synthetic", headers={"Host": "127.0.0.1:8137", "Origin": "http://127.0.0.1:8137"}, json={"count": 3})
    assert response.status_code == 200
    wrong_port = client.post("/api/synthetic", headers={"Host": "127.0.0.1:8137", "Origin": "http://127.0.0.1:9999"}, json={"count": 3})
    assert wrong_port.status_code == 403


def absent_import_record(identifier="sample", **changes):
    return {"id": f"example-{identifier}", "image_id": identifier, "image_path": f"images/{identifier}.png",
            "width": 20, "height": 10, "instruction": "Find a missing button", "target_present": False,
            "class_id": None, "bbox": None, "click_point": None, "status": "reviewed", "ambiguous": False,
            "group": identifier, "dataset_version": None, **changes}


def image_bytes(color="navy"):
    output = io.BytesIO()
    Image.new("RGB", (20, 10), color).save(output, "PNG")
    return output.getvalue()


def bundle_with_images(records, images, manifest=None):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest if manifest is not None else {"classes": dataset.DEFAULT_CLASSES}))
        archive.writestr("records.jsonl", "".join(json.dumps(record) + "\n" for record in records))
        for name, content in images.items():
            archive.writestr(name, content)
    return output.getvalue()


@pytest.mark.parametrize("malformed", [None, [], "not an object", {"status": []}, {"id": []}, {"image_id": []}])
def test_malformed_jsonl_is_a_validation_error_before_writes(client, malformed):
    malformed = {**absent_import_record(), **malformed} if isinstance(malformed, dict) else malformed
    raw = bundle_with_images([absent_import_record("first"), malformed], {"images/first.png": image_bytes(), "images/sample.png": image_bytes()})
    response = client.post("/api/datasets/import", files={"file": ("malformed.zip", raw, "application/zip")})
    assert response.status_code == 422, response.text
    assert client.get("/api/images").json() == []


def test_nonobject_manifest_is_a_validation_error(client):
    raw = bundle_with_images([absent_import_record()], {"images/sample.png": image_bytes()}, manifest=[])
    response = client.post("/api/datasets/import", files={"file": ("manifest.zip", raw, "application/zip")})
    assert response.status_code == 422 and "manifest" in response.text
    assert client.get("/api/images").json() == []


def test_oversized_later_import_image_does_not_commit_earlier_image(client):
    # PNG permits trailing bytes. This remains decodable but exceeds the per-image limit,
    # while ZIP compression keeps the test fixture small on disk and over HTTP.
    oversized = image_bytes() + b"\0" * (40 * 1024 * 1024)
    raw = bundle_with_images([absent_import_record("first"), absent_import_record("large")],
                             {"images/first.png": image_bytes(), "images/large.png": oversized})
    response = client.post("/api/datasets/import", files={"file": ("oversized.zip", raw, "application/zip")})
    assert response.status_code == 422 and "40 MB" in response.text
    assert client.get("/api/images").json() == []
    assert list((storage.ROOT / "images").iterdir()) == []


def test_windows_case_and_reserved_image_ids_preserve_independent_files(client):
    ids = ["Photo", "photo", "CON"]
    pixels = {identifier: image_bytes(color) for identifier, color in zip(ids, ("red", "green", "blue"))}
    raw = bundle_with_images([absent_import_record(identifier) for identifier in ids],
                             {f"images/{identifier}.png": pixels[identifier] for identifier in ids})
    response = client.post("/api/datasets/import", files={"file": ("portable.zip", raw, "application/zip")})
    assert response.status_code == 200, response.text
    images = client.get("/api/images").json()
    assert {im["id"] for im in images} == set(ids)
    assert len({im["image_path"].casefold() for im in images}) == 3
    for im in images:
        assert (storage.ROOT / im["image_path"]).read_bytes() == pixels[im["id"]]
    version = create_version(client, "Portable ID snapshot")
    assert len({r["image_path"].casefold() for r in version["records"]}) == 3
    assert dataset.verify_version(version["id"])
    for record in version["records"]:
        assert (storage.ROOT / record["image_path"]).read_bytes() == pixels[record["image_id"]]


@pytest.mark.parametrize("image_id", ["../outside", "space name", "é", "x" * 101])
def test_nonportable_import_image_id_is_rejected_before_commit(client, image_id):
    record = absent_import_record(image_id, image_path="images/sample.png")
    raw = bundle_with_images([absent_import_record("first"), record], {"images/first.png": image_bytes(), "images/sample.png": image_bytes()})
    response = client.post("/api/datasets/import", files={"file": ("ids.zip", raw, "application/zip")})
    assert response.status_code == 422 and "image_id" in response.text
    assert client.get("/api/images").json() == []

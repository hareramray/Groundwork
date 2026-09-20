"""Capture ZIP imports preserve grouped, unannotated screenshots atomically."""
from __future__ import annotations

import io
import json
import sqlite3
import struct
import warnings
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


def png(width=320, height=480, color="navy"):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), color).save(stream, "PNG")
    return stream.getvalue()


def capture(identifier="mobile", width=320, height=480):
    return {"id": identifier, "image_path": f"images/{identifier}-{width}x{height}.png",
            "width": width, "height": height, "viewport_width": width, "viewport_height": height,
            "device_scale_factor": 1, "url": "http://127.0.0.1:9999/responsive",
            "title": "Responsive fixture", "captured_at": "2026-09-20T00:00:00.000Z",
            "scroll_x": 0, "scroll_y": 0}


def manifest(captures=None, **changes):
    return {"format": "groundwork-captures", "schema_version": 1, "id": "batch-a",
            "name": "Responsive views", "group": "responsive-source",
            "created_at": "2026-09-20T00:00:00.000Z", "captures": captures or [capture()], **changes}


def bundle(metadata, images=None, extra=None):
    if images is None:
        images = {row["image_path"]: png(row["width"], row["height"])
                  for row in metadata["captures"]}
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(metadata))
        for name, content in images.items():
            archive.writestr(name, content)
        if extra:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr(*extra)
    return stream.getvalue()


def upload(client, raw):
    return client.post("/api/captures/import", files={"file": ("captures.zip", raw, "application/zip")})


def assert_nothing_imported(client):
    assert client.get("/api/images").json() == []
    assert list((storage.ROOT / "images").iterdir()) == []
    assert dataset.records_from_images() == []


def test_capture_import_preserves_pixels_group_and_needs_user_annotations(client):
    rows = [capture(), capture("desktop", 1024, 768)]
    metadata = manifest(rows)
    pixels = {rows[0]["image_path"]: png(color="red"),
              rows[1]["image_path"]: png(1024, 768, "blue")}
    response = upload(client, bundle(metadata, pixels))
    assert response.status_code == 200, response.text
    assert response.json()["images"] == 2
    assert response.json()["batch_id"] == metadata["id"]
    assert response.json()["group"] == metadata["group"]
    images = client.get("/api/images").json()
    assert len(images) == 2
    for image in images:
        original = next(row for row in rows if row["id"] == image["capture"]["id"])
        assert image["capture"] == original
        assert image["capture_batch_id"] == metadata["id"]
        assert image["group"] == metadata["group"]
        assert image["status"] == "unannotated"
        assert image["elements"] == [] and image["examples"] == []
        assert image["synthetic"] is False
        expected_color = (255, 0, 0) if image["width"] == 320 else (0, 0, 255)
        with Image.open(storage.ROOT / image["image_path"]) as saved:
            assert saved.size == (image["width"], image["height"])
            assert saved.getpixel((0, 0)) == expected_color
    assert dataset.records_from_images() == []
    assert client.post("/api/versions", json={"name": "Cannot train unannotated captures"}).status_code == 422


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(format="wrong-format"),
    lambda value: value.update(schema_version=999),
    lambda value: value.update(group=""),
    lambda value: value.update(captures=[]),
    lambda value: value["captures"][1].update(width=999),
    lambda value: value["captures"][1].update(width=True),
    lambda value: value["captures"][1].update(viewport_width=999),
    lambda value: value["captures"][1].update(device_scale_factor=2),
    lambda value: value["captures"][1].update(id="mobile"),
    lambda value: value["captures"][1].update(url="javascript:alert(1)"),
    lambda value: value["captures"].__setitem__(1, None),
])
def test_invalid_later_capture_never_partially_imports(client, mutation):
    metadata = manifest([capture(), capture("desktop", 1024, 768)])
    pixels = {row["image_path"]: png(row["width"], row["height"]) for row in metadata["captures"]}
    mutation(metadata)
    response = upload(client, bundle(metadata, pixels))
    assert response.status_code == 422, response.text
    assert_nothing_imported(client)


@pytest.mark.parametrize("path", ["../outside.png", "/absolute.png", "images\\escape.png", "C:/outside.png"])
def test_capture_zip_rejects_unsafe_extra_paths_before_writes(client, path):
    response = upload(client, bundle(manifest(), extra=(path, b"unsafe")))
    assert response.status_code == 422, response.text
    assert_nothing_imported(client)
    assert not (storage.ROOT.parent / "outside.png").exists()


def test_capture_zip_rejects_missing_and_duplicate_files(client):
    metadata = manifest()
    for raw in (bundle(metadata, images={}),
                bundle(metadata, extra=(metadata["captures"][0]["image_path"], png()))):
        response = upload(client, raw)
        assert response.status_code == 422, response.text
        assert_nothing_imported(client)


def test_capture_zip_rejects_corruption_nonobject_metadata_and_expansion_bomb(client):
    malformed = bundle([], images={})
    expanded = bytearray(bundle(manifest()))
    central_header = expanded.index(b"PK\x01\x02")
    struct.pack_into("<I", expanded, central_header + 24, 600 * 1024 * 1024)
    for raw in (b"not a ZIP", malformed, bytes(expanded)):
        response = upload(client, raw)
        assert response.status_code == 422, response.text
        assert_nothing_imported(client)


def test_capture_zip_limits_batch_count_and_total_pixels(client):
    too_many = [capture(f"view-{index}") for index in range(9)]
    metadata = manifest(too_many)
    response = upload(client, bundle(metadata))
    assert response.status_code == 422, response.text
    assert_nothing_imported(client)

    huge = manifest([capture(f"large-{index}", 4096, 4096) for index in range(3)])
    large_png = png(4096, 4096)
    raw = bundle(huge, {row["image_path"]: large_png for row in huge["captures"]})
    response = upload(client, raw)
    assert response.status_code == 422 and "40 megapixels" in response.text
    assert_nothing_imported(client)


def test_capture_zip_rejects_exif_rotation_instead_of_changing_viewport_dimensions(client):
    image = Image.new("RGB", (320, 480), "red")
    exif = Image.Exif()
    exif[274] = 6
    stream = io.BytesIO()
    image.save(stream, "PNG", exif=exif)
    metadata = manifest()
    raw = bundle(metadata, {metadata["captures"][0]["image_path"]: stream.getvalue()})
    response = upload(client, raw)
    assert response.status_code == 422, response.text
    assert_nothing_imported(client)


def test_capture_import_rolls_back_all_metadata_and_files_when_second_insert_fails(client):
    with storage.connection() as connection:
        connection.execute("""
            CREATE TRIGGER fail_second_capture BEFORE INSERT ON documents
            WHEN NEW.kind = 'image' AND (SELECT COUNT(*) FROM documents WHERE kind = 'image') >= 1
            BEGIN SELECT RAISE(ABORT, 'Simulated second capture persistence failure'); END
        """)
    metadata = manifest([capture(), capture("desktop", 1024, 768)])
    with pytest.raises(sqlite3.IntegrityError, match="Simulated second capture persistence failure"):
        upload(client, bundle(metadata))
    assert_nothing_imported(client)

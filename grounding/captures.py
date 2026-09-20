"""Import unannotated, grouped viewport captures from the local extension."""
from __future__ import annotations

import io
import json
import math
import re
import stat
import threading
import zipfile
import zlib
from datetime import datetime
from contextlib import closing
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from . import storage

MAX_ARCHIVE_BYTES = 250 * 1024 * 1024
MAX_IMAGE_BYTES = 40 * 1024 * 1024
IMPORT_LOCK = threading.Lock()


def _text(value, name: str, limit: int, *, empty: bool = False) -> str:
    if not isinstance(value, str) or len(value) > limit or (not empty and not value.strip()):
        raise ValueError(f'{name} must be text of {0 if empty else 1} to {limit} characters')
    try:
        value.encode('utf-8')
    except UnicodeError as error:
        raise ValueError(f'{name} must contain valid Unicode text') from error
    return value.strip()


def _identifier(value, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', value):
        raise ValueError(f'{name} must be an identifier of 1 to 100 letters, digits, underscores or hyphens')
    return value


def _timestamp(value, name: str) -> str:
    value = _text(value, name, 80)
    try:
        parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            raise ValueError('Missing time zone')
    except ValueError as error:
        raise ValueError(f'{name} must be an ISO timestamp with a time zone') from error
    return value


def _member_path(name: str) -> str:
    parts = name.rstrip('/').split('/')
    if (not name or '\\' in name or ':' in name or
            any(not part or part in {'.', '..'} for part in parts) or
            any(ord(character) < 32 for character in name)):
        raise ValueError('Unsafe capture ZIP path')
    return name


def _validate(archive: zipfile.ZipFile) -> tuple[dict, list[tuple[dict, bytes]]]:
    infos = archive.infolist()
    if len(infos) > 64 or sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES:
        raise ValueError('Expanded capture ZIP exceeds 250 MB or 64 entries')
    if len({info.filename.casefold() for info in infos}) != len(infos):
        raise ValueError('Capture ZIP entries must have unique names')
    for info in infos:
        _member_path(info.filename)
        if info.flag_bits & 1 or stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('Capture ZIP cannot contain encrypted files or symbolic links')
    files = {info.filename: info for info in infos if not info.is_dir()}
    if 'manifest.json' not in files or files['manifest.json'].file_size > 1024 * 1024:
        raise ValueError('Capture ZIP requires manifest.json, at most 1 MB')
    try:
        manifest = json.loads(archive.read('manifest.json').decode('utf-8-sig'))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError('Capture manifest must be valid UTF-8 JSON') from error
    if (not isinstance(manifest, dict) or manifest.get('format') != 'groundwork-captures' or
            type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1):
        raise ValueError('Choose a Groundwork capture ZIP (format groundwork-captures, schema version 1)')
    batch_id = _identifier(manifest.get('id'), 'Capture batch ID')
    group = _text(manifest.get('group'), 'Capture group', 500)
    _text(manifest.get('name', 'Webpage captures'), 'Capture name', 160)
    _timestamp(manifest.get('created_at'), 'Capture batch created_at')
    captures = manifest.get('captures')
    if not isinstance(captures, list) or not 1 <= len(captures) <= 8:
        raise ValueError('Capture ZIP must contain 1 to 8 snapshots')
    prepared, seen_ids, seen_paths, total_pixels = [], set(), set(), 0
    for capture in captures:
        if not isinstance(capture, dict):
            raise ValueError('Each capture must be a JSON object')
        capture_id = _identifier(capture.get('id'), 'Capture ID')
        if capture_id in seen_ids:
            raise ValueError('Capture IDs must be unique within a batch')
        seen_ids.add(capture_id)
        path = capture.get('image_path')
        if (not isinstance(path, str) or path not in files or
                not path.startswith('images/') or PurePosixPath(path).suffix.lower() != '.png'):
            raise ValueError('Each capture must reference a PNG in the ZIP images/ folder')
        if path in seen_paths:
            raise ValueError('Each capture must reference its own PNG file')
        seen_paths.add(path)
        if files[path].file_size > MAX_IMAGE_BYTES:
            raise ValueError('A capture image exceeds 40 MB')
        for field in ('width', 'height', 'viewport_width', 'viewport_height'):
            if type(capture.get(field)) is not int or not 240 <= capture[field] <= 4096:
                raise ValueError(f'Capture {field} must be an integer between 240 and 4096')
        if (capture['width'] != capture['viewport_width'] or capture['height'] != capture['viewport_height'] or
                type(capture.get('device_scale_factor')) not in (int, float) or capture['device_scale_factor'] != 1):
            raise ValueError('Capture dimensions must match its viewport at device scale factor 1')
        url = _text(capture.get('url'), 'Capture URL', 8192)
        try:
            parsed = urlsplit(url)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname:
                raise ValueError('Unsupported URL')
        except ValueError as error:
            raise ValueError('Capture URL must be an HTTP or HTTPS webpage') from error
        _text(capture.get('title', ''), 'Capture title', 1000, empty=True)
        _timestamp(capture.get('captured_at'), 'Capture captured_at')
        for field in ('scroll_x', 'scroll_y'):
            number = capture.get(field)
            if type(number) not in (int, float) or not math.isfinite(number) or not 0 <= number <= 10_000_000:
                raise ValueError(f'Capture {field} must be a finite, nonnegative number')
        total_pixels += capture['width'] * capture['height']
        if total_pixels > 40_000_000:
            raise ValueError('Capture batch exceeds 40 megapixels')
        raw = archive.read(path)
        try:
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.size != (capture['width'], capture['height']):
                    raise ValueError('Capture PNG dimensions do not match the manifest')
                image.verify()
            # Decode now, before committing any earlier image in this batch.
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.getexif().get(274, 1) != 1:
                    raise ValueError('Capture PNGs must not contain EXIF rotation or mirroring')
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
            raise ValueError('Capture ZIP contains an unreadable PNG') from error
        prepared.append((dict(capture), raw))
    if set(files) != {'manifest.json', *seen_paths}:
        raise ValueError('Capture ZIP contains files not referenced by the manifest')
    return {'id': batch_id, 'group': group}, prepared


def import_captures(content: bytes) -> dict:
    if len(content) > MAX_ARCHIVE_BYTES:
        raise ValueError('Capture ZIP exceeds 250 MB')
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            manifest, prepared = _validate(archive)
    except (zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as error:
        raise ValueError('Choose a valid, unencrypted capture ZIP exported by Groundwork') from error
    created, paths = [], []
    with IMPORT_LOCK:
        try:
            for capture, raw in prepared:
                image_id = storage.uid()
                relative = f'images/{image_id}.png'
                path = storage.safe_path(relative)
                item = {'id': image_id, 'filename': PurePosixPath(capture['image_path']).name,
                        'width': capture['width'], 'height': capture['height'], 'image_path': relative,
                        'url': f'/api/files/{relative}', 'group': manifest['group'],
                        'status': 'unannotated', 'synthetic': False, 'elements': [], 'examples': [],
                        'created_at': storage.now(), 'capture_batch_id': manifest['id'], 'capture': capture}
                # Serialize before touching the filesystem, including any extra metadata.
                encoded = json.dumps(item, allow_nan=False, ensure_ascii=False)
                encoded.encode('utf-8')
                paths.append(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                created.append((item, encoded))
            # Publish all image records together, after every PNG has been written.
            with closing(storage.connection()) as conn, conn:
                conn.executemany('INSERT INTO documents(kind,id,data,created_at) VALUES (?,?,?,?)',
                                 [('image', item['id'], encoded, item['created_at']) for item, encoded in created])
        except Exception:
            # The DB transaction rolls back; also remove complete or partially written PNGs.
            for path in paths:
                path.unlink(missing_ok=True)
            raise
    return {'images': len(created), 'batch_id': manifest['id'], 'group': manifest['group']}

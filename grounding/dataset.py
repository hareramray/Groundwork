"""Annotation validation, immutable snapshots, leakage-safe splitting and JSONL exchange."""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps

from . import storage as s

DEFAULT_CLASSES = ['button', 'textbox', 'link', 'checkbox', 'dropdown', 'icon']
STATUSES = {'draft', 'reviewed', 'excluded'}


def classes():
    try:
        return s.get('settings', 'classes')['classes']
    except KeyError:
        return DEFAULT_CLASSES.copy()


def set_classes(values):
    if not isinstance(values, list) or not values or len(values) > 128:
        raise ValueError('Provide 1 to 128 unique class names')
    if any(not isinstance(v, str) or not v.strip() or len(v) > 80 for v in values):
        raise ValueError('Class names must be nonempty strings up to 80 characters')
    values = [v.strip() for v in values]
    if len(set(values)) != len(values):
        raise ValueError('Class names must be unique')
    old = classes()
    used = {e['class_id'] for im in s.list_items('image') for e in im.get('elements', [])}
    if any(c >= len(values) or values[c] != old[c] for c in used):
        raise ValueError('Existing annotations use these class IDs. Keep their names and positions; append new classes instead.')
    return s.put('settings', 'classes', {'classes': values})


def valid_box(box):
    return isinstance(box, (list, tuple)) and len(box) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box) and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1


def valid_point(point, box):
    return isinstance(point, (list, tuple)) and len(point) == 2 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in point) and box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def image_status(examples, elements):
    if not examples:
        return 'draft' if elements else 'unannotated'
    if all(e['status'] == 'excluded' for e in examples):
        return 'excluded'
    if all(e['status'] in ('reviewed', 'excluded') and not e.get('ambiguous') for e in examples):
        return 'reviewed'
    return 'draft'


def validate_annotations(elements, examples, class_names=None):
    names = class_names or classes()
    errors = []
    ids = set()
    for e in elements:
        eid = e.get('id')
        if not isinstance(eid, str) or not eid or eid in ids:
            errors.append('Elements need unique nonempty IDs')
        if isinstance(eid, str):
            ids.add(eid)
        cid = e.get('class_id')
        if not isinstance(cid, int) or isinstance(cid, bool) or not 0 <= cid < len(names):
            errors.append(f'Element {eid}: unknown class')
        if not valid_box(e.get('bbox')):
            errors.append(f'Element {eid}: invalid normalized bounding box')
        elif not valid_point(e.get('click_point'), e['bbox']):
            errors.append(f'Element {eid}: click point must lie inside its target box')
    example_ids = set()
    for ex in examples:
        eid = ex.get('id')
        if not isinstance(eid, str) or not eid or eid in example_ids:
            errors.append('Examples need unique nonempty IDs')
        if isinstance(eid, str):
            example_ids.add(eid)
        if not isinstance(ex.get('instruction'), str) or not ex['instruction'].strip():
            errors.append(f'Example {eid}: instruction is required')
        elif len(ex['instruction']) > 2000:
            errors.append(f'Example {eid}: instruction exceeds 2000 characters')
        if not isinstance(ex.get('status'), str) or ex['status'] not in STATUSES:
            errors.append(f'Example {eid}: status must be draft, reviewed or excluded')
        if not isinstance(ex.get('target_present'), bool):
            errors.append(f'Example {eid}: target_present must be boolean')
        elif ex['target_present'] and (not isinstance(ex.get('element_id'), str) or ex['element_id'] not in ids):
            errors.append(f'Example {eid}: select a matching element')
        elif not ex['target_present'] and ex.get('element_id') is not None:
            errors.append(f'Example {eid}: absent examples must have null targets')
        if ex.get('ambiguous') and ex.get('status') == 'reviewed':
            errors.append(f'Example {eid}: resolve ambiguity before marking reviewed')
    return errors


def save_annotations(image_id, payload):
    im = s.get('image', image_id)
    elements = payload.get('elements', [])
    examples = payload.get('examples', [])
    if not isinstance(elements, list) or not isinstance(examples, list) or any(not isinstance(x, dict) for x in elements + examples):
        raise ValueError('elements and examples must be arrays of objects')
    errors = validate_annotations(elements, examples)
    if errors:
        raise ValueError('; '.join(errors))
    other_ids = {ex['id'] for other in s.list_items('image') if other['id'] != image_id for ex in other['examples']}
    if any(ex['id'] in other_ids for ex in examples):
        raise ValueError('Example IDs must be globally unique')
    group = payload.get('group', im.get('group', ''))
    if not isinstance(group, str) or len(group) > 500:
        raise ValueError('Group must be a string up to 500 characters')
    im.update(elements=elements, examples=examples, group=group.strip(), status=image_status(examples, elements), updated_at=s.now())
    return s.put('image', image_id, im)


def add_image(content, filename, *, image_id=None):
    if len(content) > 40 * 1024 * 1024:
        raise ValueError('Image exceeds 40 MB')
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError('Image exceeds 40 megapixels')
            image = ImageOps.exif_transpose(source).convert('RGB')
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError('Upload a valid PNG, JPEG or WebP image') from exc
    image_id = image_id or s.uid()
    # Metadata IDs and filenames are independent: imported Windows reserved names,
    # case-insensitive filesystems and user-provided IDs cannot alias image bytes.
    file_id = s.uid()
    path = s.ROOT / 'images' / f'{file_id}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, 'PNG')
    im = {'id': image_id, 'filename': Path(filename).name, 'width': image.width, 'height': image.height, 'image_path': path.relative_to(s.ROOT).as_posix(), 'url': f'/api/files/images/{file_id}.png', 'group': '', 'status': 'unannotated', 'synthetic': False, 'elements': [], 'examples': [], 'created_at': s.now()}
    return s.put('image', image_id, im)


def records_from_images():
    records = []
    for im in s.list_items('image'):
        targets = {e['id']: e for e in im['elements']}
        for ex in im['examples']:
            target = targets.get(ex.get('element_id')) if ex.get('target_present') else None
            records.append({'id': ex['id'], 'image_id': im['id'], 'image_path': im['image_path'], 'width': im['width'], 'height': im['height'], 'instruction': ex['instruction'], 'target_present': ex['target_present'], 'class_id': target['class_id'] if target else None, 'bbox': target['bbox'] if target else None, 'click_point': target['click_point'] if target else None, 'status': ex['status'], 'ambiguous': ex.get('ambiguous', False), 'group': im.get('group', ''), 'synthetic': im.get('synthetic', False), 'dataset_version': None})
    return records


def record_errors(record, names):
    if not isinstance(record, dict):
        return ['Each JSONL record must be an object']
    errors = []
    for key in ('id', 'image_id', 'image_path', 'instruction'):
        if not isinstance(record.get(key), str) or not record[key].strip():
            errors.append(f'{key} must be a nonempty string')
    if isinstance(record.get('instruction'), str) and len(record['instruction']) > 2000:
        errors.append('Instruction exceeds 2000 characters')
    if isinstance(record.get('image_id'), str) and not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', record['image_id']):
        errors.append('image_id must contain 1 to 100 ASCII letters, digits, underscores or hyphens')
    for key in ('width', 'height'):
        if not isinstance(record.get(key), int) or isinstance(record.get(key), bool) or record[key] <= 0:
            errors.append(f'{key} must be a positive integer')
    if not isinstance(record.get('status'), str) or record['status'] not in STATUSES:
        errors.append('Unknown review status')
    if not isinstance(record.get('target_present'), bool):
        errors.append('target_present must be boolean')
    elif record['target_present']:
        cid = record.get('class_id')
        if not isinstance(cid, int) or isinstance(cid, bool) or not 0 <= cid < len(names):
            errors.append('Unknown class ID')
        if not valid_box(record.get('bbox')):
            errors.append('Invalid bounding box')
        elif not valid_point(record.get('click_point'), record['bbox']):
            errors.append('Click point is outside target')
    elif any(record.get(k) is not None for k in ('class_id', 'bbox', 'click_point')):
        errors.append('Absent examples require null class_id, bbox and click_point')
    if record.get('ambiguous') and record.get('status') == 'reviewed':
        errors.append('Ambiguous example cannot be reviewed')
    if not isinstance(record.get('group', ''), str) or len(record.get('group', '')) > 500:
        errors.append('group must be a string up to 500 characters')
    return errors


def statistics(records):
    return {'examples': len(records), 'images': len({r['image_id'] for r in records}), 'class_distribution': dict(Counter(str(r['class_id']) for r in records if r['target_present'])), 'absent_targets': sum(not r['target_present'] for r in records), 'unreviewed': sum(r['status'] != 'reviewed' for r in records), 'ambiguous': sum(bool(r.get('ambiguous')) for r in records), 'excluded': sum(r['status'] == 'excluded' for r in records), 'synthetic': sum(bool(r.get('synthetic')) for r in records), 'split_sizes': dict(Counter(r.get('split', 'unsplit') for r in records))}


def validate_dataset(group_by='group', seed=42):
    if group_by not in ('group', 'image'):
        raise ValueError('group_by must be group or image')
    records = records_from_images()
    names = classes()
    errors, warnings = [], []
    seen = set()
    checked_images = set()
    for r in records:
        errors.extend(f"{r['id']}: {e}" for e in record_errors(r, names))
        if r['id'] in seen:
            errors.append(f"Duplicate example ID {r['id']}")
        seen.add(r['id'])
        path = s.safe_path(r['image_path'])
        if not path.is_file():
            errors.append(f"Missing image {r['image_id']}")
        elif r['image_id'] not in checked_images:
            try:
                with Image.open(path) as source:
                    if source.size != (r['width'], r['height']):
                        errors.append(f"Image dimensions changed for {r['image_id']}")
                    source.verify()
            except OSError:
                errors.append(f"Unreadable image {r['image_id']}")
            checked_images.add(r['image_id'])
    eligible = [r for r in records if r['status'] == 'reviewed' and not r.get('ambiguous')]
    if not eligible:
        errors.append('No reviewed, unambiguous examples. Explicitly review annotations first.')
    if len(eligible) < 100:
        warnings.append('Fewer than 100 eligible examples; useful for workflow checks, insufficient evidence of real-world quality.')
    if any(r['status'] != 'reviewed' or r.get('ambiguous') for r in records):
        warnings.append('Draft, ambiguous and excluded examples will not enter the snapshot.')
    if group_by == 'image':
        warnings.append('Image-only splitting does not protect related website/template/session images. Prefer group splitting.')
    elif any(not r['group'] for r in eligible):
        warnings.append('Some images have no website/template/session group; they are grouped by screenshot only.')
    if not any(not r['target_present'] for r in eligible):
        warnings.append('No absent-target examples; presence decisions need both positive and absent examples.')
    used = {r['class_id'] for r in eligible if r['target_present']}
    if len(used) < len(names):
        warnings.append('Some configured classes have no reviewed positive examples.')
    groups = {r['group'] if group_by == 'group' and r['group'] else r['image_id'] for r in eligible}
    if len(groups) < 3:
        warnings.append('Fewer than three independent groups; validation or test split will be empty.')
    stats = statistics(records)
    stats.update(eligible=len(eligible), groups=len(groups))
    return {'errors': errors, 'warnings': warnings, 'stats': stats}


def assign_splits(records, seed, group_by, train_ratio, val_ratio):
    """Union related images AND identical image bytes; never split a screenshot."""
    parents = {}
    def find(x):
        parents.setdefault(x, x)
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]
    def union(a, b):
        parents[find(a)] = find(b)
    hashes = {}
    for r in records:
        image = 'image:' + r['image_id']
        find(image)
        if group_by == 'group' and r.get('group'):
            union(image, 'group:' + r['group'])
        if r['image_id'] not in hashes:
            hashes[r['image_id']] = hashlib.sha256(s.safe_path(r['image_path']).read_bytes()).hexdigest()
        union(image, 'hash:' + hashes[r['image_id']])
    groups = sorted({find('image:' + r['image_id']) for r in records})
    random.Random(seed).shuffle(groups)
    n = len(groups)
    nval = max(1 if n >= 3 and val_ratio > 0 else 0, round(n * val_ratio))
    ntest = max(1 if n >= 3 and 1 - train_ratio - val_ratio > 1e-8 else 0, round(n * (1 - train_ratio - val_ratio)))
    while nval + ntest >= n and nval + ntest > 0:
        if ntest >= nval:
            ntest -= 1
        else:
            nval -= 1
    mapping = {g: ('val' if i < nval else 'test' if i < nval + ntest else 'train') for i, g in enumerate(groups)}
    return [dict(r, split=mapping[find('image:' + r['image_id'])]) for r in records]


def fingerprint(manifest):
    value = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def create_version(payload):
    name = str(payload.get('name', '')).strip()
    if not name:
        raise ValueError('Dataset version name is required')
    group_by = payload.get('group_by', 'group')
    seed = int(payload.get('seed', 42))
    train_ratio, val_ratio = float(payload.get('train_ratio', .7)), float(payload.get('val_ratio', .15))
    if not 0 < train_ratio <= 1 or not 0 <= val_ratio < 1 or train_ratio + val_ratio > 1:
        raise ValueError('Split ratios must be nonnegative, sum to at most one, with a positive training ratio')
    report = validate_dataset(group_by, seed)
    if report['errors']:
        raise ValueError('; '.join(report['errors']))
    records = [r for r in records_from_images() if r['status'] == 'reviewed' and not r.get('ambiguous')]
    records = assign_splits(records, seed, group_by, train_ratio, val_ratio)
    id = s.uid()
    destination = s.ROOT / 'versions' / id
    staging = s.ROOT / 'versions' / ('.' + id)
    (staging / 'images').mkdir(parents=True)
    hashes = {}
    try:
        for r in records:
            image_name = hashlib.sha256(r['image_id'].encode('utf-8')).hexdigest() + '.png'
            path = staging / 'images' / image_name
            if not path.exists():
                shutil.copyfile(s.safe_path(r['image_path']), path)
            r['image_path'] = f'versions/{id}/images/{image_name}'
            r['dataset_version'] = id
            hashes[r['image_path']] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {'id': id, 'name': name, 'created_at': s.now(), 'schema_version': 1, 'classes': classes(), 'seed': seed, 'group_by': group_by, 'train_ratio': train_ratio, 'val_ratio': val_ratio, 'records': records, 'stats': statistics(records), 'image_hashes': hashes, 'warnings': report['warnings']}
        manifest['fingerprint'] = fingerprint(manifest)
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (staging / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records), encoding='utf-8')
        staging.rename(destination)
        s.put('version', id, {k: v for k, v in manifest.items() if k not in ('records', 'image_hashes')})
        return manifest
    except Exception:
        if staging.is_dir():
            shutil.rmtree(staging)
        raise


def load_version(id):
    s.get('version', id)
    path = s.safe_path(f'versions/{id}/manifest.json')
    return json.loads(path.read_text(encoding='utf-8'))


def verify_version(id):
    manifest = load_version(id)
    expected = s.get('version', id)['fingerprint']
    if manifest.get('fingerprint') != expected or fingerprint(manifest) != expected:
        raise ValueError('Dataset snapshot metadata has changed; exact resume is incompatible. Create a new version.')
    for path, expected_hash in manifest['image_hashes'].items():
        source = s.safe_path(path)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Dataset snapshot image changed or is missing: {path}')
    return manifest


def export_version(id):
    manifest = verify_version(id)
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        copied = set()
        records = []
        for r in manifest['records']:
            relative = 'images/' + Path(r['image_path']).name
            if relative not in copied:
                archive.write(s.safe_path(r['image_path']), relative)
                copied.add(relative)
            records.append(dict(r, image_path=relative))
        archive.writestr('records.jsonl', ''.join(json.dumps(r) + '\n' for r in records))
        archive.writestr('manifest.json', json.dumps({'schema_version': 1, 'classes': manifest['classes'], 'source_version': id, 'name': manifest['name'], 'coordinate_format': 'normalized_xyxy'}, indent=2))
    return output.getvalue()


def import_dataset(content):
    """Validate entire ZIP before committing. Existing IDs cannot overwrite user work."""
    if len(content) > 250 * 1024 * 1024:
        raise ValueError('Import exceeds 250 MB')
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise ValueError('Expected a ZIP containing manifest.json, records.jsonl and images/') from exc
    with archive:
        infos = archive.infolist()
        if sum(i.file_size for i in infos) > 500 * 1024 * 1024 or len(infos) > 10000:
            raise ValueError('Expanded import exceeds 500 MB or 10000 files')
        if len({info.filename for info in infos}) != len(infos):
            raise ValueError('ZIP entries must have unique names')
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename:
                raise ValueError('Unsafe ZIP path')
        if 'manifest.json' not in archive.namelist() or 'records.jsonl' not in archive.namelist():
            raise ValueError('ZIP needs manifest.json and records.jsonl')
        manifest = json.loads(archive.read('manifest.json'))
        if not isinstance(manifest, dict):
            raise ValueError('manifest.json must contain an object')
        if manifest.get('classes') != classes():
            raise ValueError('Import class mapping differs. Configure compatible class names and order first.')
        records = [json.loads(line) for line in archive.read('records.jsonl').decode('utf-8-sig').splitlines() if line.strip()]
        if not records:
            raise ValueError('Import has no records')
        known = {r['id'] for r in records_from_images()}
        seen, images, grouped = set(), {}, {}
        for r in records:
            errors = record_errors(r, classes())
            if errors:
                raise ValueError(f"Record {r.get('id') if isinstance(r, dict) else '(invalid)'}: {'; '.join(errors)}")
            if r['id'] in seen or r['id'] in known:
                raise ValueError(f"Duplicate existing example ID {r['id']}; import never overwrites annotations")
            seen.add(r['id'])
            path = r['image_path']
            if path not in archive.namelist() or not path.startswith('images/'):
                raise ValueError('Records must reference images/ files present in the ZIP')
            if r['image_id'] not in images:
                raw = archive.read(path)
                if len(raw) > 40 * 1024 * 1024:
                    raise ValueError('An imported image exceeds 40 MB')
                try:
                    with Image.open(io.BytesIO(raw)) as source:
                        if source.width * source.height > 40_000_000:
                            raise ValueError('An imported image exceeds 40 megapixels')
                        im = ImageOps.exif_transpose(source)
                        if im.size != (r['width'], r['height']):
                            raise ValueError('Original dimensions do not match the imported image')
                        im.load()
                except (OSError, Image.DecompressionBombError) as exc:
                    raise ValueError('Import contains an unreadable or unsafe image') from exc
                images[r['image_id']] = (raw, path, r['width'], r['height'], r.get('group', ''))
            elif images[r['image_id']][1:] != (path, r['width'], r['height'], r.get('group', '')):
                raise ValueError('All records for an image must agree on path, dimensions and group')
            grouped.setdefault(r['image_id'], []).append(r)
        # All records/images validated before any persistent changes.
        count = 0
        for image_id, (raw, path, _, _, group) in images.items():
            try:
                existing = s.get('image', image_id)
                if hashlib.sha256(s.safe_path(existing['image_path']).read_bytes()).digest() != hashlib.sha256(raw).digest():
                    raise ValueError(f'Image ID {image_id} already exists with different contents')
            except KeyError:
                existing = None
            # Existing image IDs are permitted to add distinct examples to the same screenshot.
            if existing and existing.get('group', '') != group:
                raise ValueError(f'Image ID {image_id} already exists with a different group')
        for image_id, (raw, path, _, _, group) in images.items():
            try:
                im = s.get('image', image_id)
            except KeyError:
                im = add_image(raw, Path(path).name, image_id=image_id)
                count += 1
            elements, examples = list(im['elements']), list(im['examples'])
            for r in grouped[image_id]:
                element_id = s.uid() if r['target_present'] else None
                if element_id:
                    elements.append({'id': element_id, 'class_id': r['class_id'], 'label': '', 'bbox': r['bbox'], 'click_point': r['click_point']})
                examples.append({'id': r['id'], 'instruction': r['instruction'], 'target_present': r['target_present'], 'element_id': element_id, 'status': r['status'], 'ambiguous': r.get('ambiguous', False)})
            save_annotations(im['id'], {'elements': elements, 'examples': examples, 'group': group})
            if any(r.get('synthetic') for r in grouped[image_id]):
                s.patch('image', im['id'], {'synthetic': True})
    return {'images': count, 'examples': len(records), 'warnings': ['Imported split labels are provenance only. A new immutable version computes splits across all eligible local data.']}

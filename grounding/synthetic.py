"""Deterministic browser-like samples, solely for workflow verification."""
import io
import random
from PIL import Image, ImageDraw
from . import dataset, storage as s


def generate(count=24, seed=42, reviewed=False):
    if not 3 <= count <= 500:
        raise ValueError('Synthetic image count must be between 3 and 500')
    rng = random.Random(seed)
    names = dataset.classes()
    total = 0
    for index in range(count):
        w, h = 640, 400
        image = Image.new('RGB', (w, h), '#f4f6fa')
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, w, 38), fill='#dce3ee')
        draw.rounded_rectangle((84, 8, 580, 30), 7, fill='white')
        draw.text((95, 13), f'SYNTHETIC / sample-{seed}-{index}.local', fill='#45546c')
        draw.text((24, 57), 'Synthetic browser workspace', fill='#1d2b41')
        elements = []
        used_cells = rng.sample(range(12), min(4, len(names)))
        selected_classes = rng.sample(range(len(names)), min(4, len(names)))
        for j, (cell, cid) in enumerate(zip(used_cells, selected_classes)):
            x = 25 + (cell % 4) * 154 + rng.randint(0, 8)
            y = 100 + (cell // 4) * 94 + rng.randint(0, 8)
            bw, bh = rng.randint(78, 125), rng.randint(25, 48)
            color_name, color = rng.choice([('blue', '#427be2'), ('green', '#208b75'), ('red', '#d85560')])
            label = f'{color_name} {names[cid]}'
            draw.rounded_rectangle((x, y, x + bw, y + bh), 5, fill=color)
            draw.text((x + 5, y + 7), names[cid], fill='white')
            elements.append({'id': s.uid(), 'class_id': cid, 'label': label, 'bbox': [x / w, y / h, (x + bw) / w, (y + bh) / h], 'click_point': [(x + bw / 2) / w, (y + bh / 2) / h]})
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
        im = dataset.add_image(buffer.getvalue(), f'synthetic-{seed}-{index:03d}.png')
        examples = [{'id': s.uid(), 'instruction': 'Find the ' + e['label'], 'target_present': True, 'element_id': e['id'], 'status': 'reviewed' if reviewed else 'draft', 'ambiguous': False} for e in elements]
        examples.append({'id': s.uid(), 'instruction': 'Find the purple missing control', 'target_present': False, 'element_id': None, 'status': 'reviewed' if reviewed else 'draft', 'ambiguous': False})
        im = dataset.save_annotations(im['id'], {'elements': elements, 'examples': examples, 'group': f'synthetic-session-{seed}-{index // 2}'})
        s.patch('image', im['id'], {'synthetic': True})
        total += len(examples)
    return {'images': count, 'examples': total, 'synthetic': True, 'reviewed': reviewed}

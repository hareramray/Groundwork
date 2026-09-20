"""Loopback-only local API. Training is isolated in subprocess workers."""
from __future__ import annotations

import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from . import dataset, storage as s, synthetic

# Heavy jobs can share one GPU. Serialize inference/evaluation to bound device usage.
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    s.init_db()
    from .training import recover_runs
    from .chat import recover_runs as recover_chat_runs
    recover_runs()
    recover_chat_runs()
    yield


app = FastAPI(title='Groundwork Local Grounding Lab', version='1.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origin_regex=r'http://(127\.0\.0\.1|localhost)(:\d{1,5})?', allow_credentials=False, allow_methods=['GET', 'POST', 'PUT', 'DELETE'], allow_headers=['Content-Type'])


@app.middleware('http')
async def local_origin(request: Request, call_next):
    # No remote web page may instruct a local service to mutate private datasets.
    origin = request.headers.get('origin')
    if origin:
        try:
            parsed = urlsplit(origin)
            allowed = parsed.scheme == 'http' and parsed.hostname in {'127.0.0.1', 'localhost'} and parsed.port in {request.url.port or 80, 5173}
        except ValueError:
            allowed = False
        if not allowed:
            return JSONResponse({'detail': 'Only the local Groundwork interface may access this API'}, 403)
    host = request.headers.get('host', '').split(':')[0]
    if host not in {'localhost', '127.0.0.1', 'testserver', '[::1]'}:
        return JSONResponse({'detail': 'This application is local-only'}, 403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def bad_input(request, exc):
    return JSONResponse({'detail': str(exc)}, status_code=422)


@app.exception_handler(KeyError)
async def missing(request, exc):
    return JSONResponse({'detail': str(exc).strip("'")}, status_code=404)


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ClassConfig(Strict):
    classes: list[str]


class AnnotationPayload(Strict):
    elements: list[dict]
    examples: list[dict]
    group: str = ''


class SyntheticPayload(Strict):
    count: int = Field(24, ge=3, le=500)
    seed: int = 42
    reviewed: bool = False


class ValidationPayload(Strict):
    group_by: str = 'group'
    seed: int = 42


class VersionPayload(ValidationPayload):
    name: str
    train_ratio: float = .7
    val_ratio: float = .15


class RunPayload(Strict):
    name: str = 'Untitled model'
    version_id: str
    mode: str = 'fresh'
    parent_run_id: str | None = None
    source_checkpoint: str | None = 'latest'
    source_export_id: str | None = None
    config: dict = Field(default_factory=dict)


class ExportPayload(Strict):
    checkpoint: str = 'latest'


class ChatExamplePayload(Strict):
    prompt: str = Field(min_length=1, max_length=500)
    response: str = Field(min_length=1, max_length=500)


class ChatRunPayload(Strict):
    name: str = Field('My chat model', max_length=160)
    config: dict = Field(default_factory=dict)
    source_run_id: str | None = None
    source_chat_run_id: str | None = None
    source_checkpoint: str = 'latest'


class ChatPredictionPayload(Strict):
    run_id: str
    message: str = Field(min_length=1, max_length=500)


class EvaluationPayload(ExportPayload):
    run_id: str
    version_id: str
    split: str = 'test'
    threshold: float = Field(.5, ge=0, le=1)


class CorrectionPayload(Strict):
    image_id: str
    instruction: str
    target_present: bool
    class_id: int | None = None
    bbox: list[float] | None = None
    click_point: list[float] | None = None


@app.get('/api/health')
def health():
    return {'status': 'ok', 'local_only': True, 'data_dir': str(s.ROOT)}


@app.get('/api/dashboard')
def dashboard():
    import torch
    import psutil
    import platform
    images = s.list_items('image')
    examples = [e for im in images for e in im['examples']]
    gpu = {'available': torch.cuda.is_available(), 'name': 'CPU', 'total_memory_mb': None, 'torch_version': torch.__version__, 'cuda_version': torch.version.cuda}
    if gpu['available']:
        prop = torch.cuda.get_device_properties(0)
        gpu.update(name=prop.name, total_memory_mb=round(prop.total_memory / 1024 ** 2), free_memory_mb=round(torch.cuda.mem_get_info()[0] / 1024 ** 2))
    hardware = {'cpu': platform.processor(), 'logical_cores': psutil.cpu_count(), 'ram_gb': round(psutil.virtual_memory().total / 1024 ** 3, 1), 'os': platform.system(), 'python': platform.python_version()}
    return {'images': len(images), 'examples': len(examples), 'reviewed': sum(e['status'] == 'reviewed' and not e.get('ambiguous') for e in examples), 'versions': len(s.list_items('version')), 'runs': len(s.list_items('run')), 'gpu': gpu, 'hardware': hardware}


@app.get('/api/classes')
def get_classes():
    return {'classes': dataset.classes()}


@app.put('/api/classes')
def configure_classes(payload: ClassConfig):
    return dataset.set_classes(payload.classes)


@app.get('/api/images')
def images():
    return s.list_items('image')


@app.post('/api/images')
def upload_images(files: Annotated[list[UploadFile], File()]):
    if len(files) > 100:
        raise ValueError('Upload at most 100 images at a time')
    return [dataset.add_image(file.file.read(40 * 1024 * 1024 + 1), file.filename or 'screenshot.png') for file in files]


@app.get('/api/images/{image_id}')
def get_image(image_id: str):
    return s.get('image', image_id)


@app.put('/api/images/{image_id}')
def save_image(image_id: str, payload: AnnotationPayload):
    return dataset.save_annotations(image_id, payload.model_dump())


@app.delete('/api/images/{image_id}')
def delete_image(image_id: str):
    im = s.get('image', image_id)
    # Snapshots own independent image copies; no run is modified.
    s.delete('image', image_id)
    s.safe_path(im['image_path']).unlink(missing_ok=True)
    return {'deleted': image_id}


@app.post('/api/synthetic')
def generate_synthetic(payload: SyntheticPayload):
    return synthetic.generate(**payload.model_dump())


@app.post('/api/datasets/validate')
def validate(payload: ValidationPayload):
    return dataset.validate_dataset(**payload.model_dump())


@app.post('/api/datasets/import')
def import_zip(file: Annotated[UploadFile, File()]):
    return dataset.import_dataset(file.file.read(250 * 1024 * 1024 + 1))


@app.get('/api/versions')
def versions():
    return s.list_items('version')


@app.post('/api/versions')
def create_version(payload: VersionPayload):
    return dataset.create_version(payload.model_dump())


@app.get('/api/versions/{version_id}')
def get_version(version_id: str):
    return dataset.load_version(version_id)


@app.get('/api/versions/{version_id}/export')
def export_version(version_id: str):
    return Response(dataset.export_version(version_id), media_type='application/zip', headers={'Content-Disposition': f'attachment; filename="dataset-{version_id}.zip"'})


@app.get('/api/training/defaults')
def training_defaults():
    from .training import default_config
    return default_config()


@app.get('/api/runs')
def runs():
    from .training import recover_runs
    recover_runs()
    return s.list_items('run')


@app.post('/api/runs')
def create_run(payload: RunPayload):
    from .training import create_run, launch_run
    data = payload.model_dump(exclude_none=True)
    run = create_run(data)
    try:
        return launch_run(run['id'])
    except (ValueError, OSError) as exc:
        s.patch('run', run['id'], {'status': 'error', 'error': str(exc), 'pid': None})
        raise ValueError(f'Could not launch training: {exc}') from exc


@app.get('/api/runs/{run_id}')
def get_run(run_id: str):
    from .training import recover_runs
    recover_runs()
    return s.get('run', run_id)


@app.get('/api/runs/{run_id}/checkpoints')
def checkpoints(run_id: str):
    from .training import list_checkpoints
    return list_checkpoints(run_id)


@app.post('/api/runs/{run_id}/export')
def export_model(run_id: str, payload: ExportPayload):
    from .training import export_model
    result = export_model(run_id, payload.checkpoint)
    return dict(result, download_url=f"/api/exports/{result['id']}/download")


@app.post('/api/runs/{run_id}/{action}')
def run_control(run_id: str, action: str):
    from .training import control_run
    if action not in ('pause', 'resume', 'stop'):
        raise HTTPException(404, 'Unknown training action')
    return control_run(run_id, action)


@app.get('/api/exports')
def exports():
    return [dict(e, download_url=f"/api/exports/{e['id']}/download") for e in s.list_items('export')]


@app.get('/api/exports/{export_id}/download')
def download_export(export_id: str):
    export = s.get('export', export_id)
    path = s.safe_path(export.get('path', f'exports/{export_id}.pt'))
    if not path.is_file():
        raise HTTPException(404, 'Export file missing')
    return FileResponse(path, filename=path.name, media_type='application/octet-stream')


@app.post('/api/predict')
def predict(file: Annotated[UploadFile, File()], instruction: Annotated[str, Form()], run_id: Annotated[str, Form()], checkpoint: Annotated[str, Form()] = 'latest', threshold: Annotated[float, Form()] = .5):
    from .inference import predict as run_prediction
    if not instruction.strip() or len(instruction) > 2000:
        raise ValueError('Provide an instruction of 1 to 2000 characters')
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one')
    im = dataset.add_image(file.file.read(40 * 1024 * 1024 + 1), file.filename or 'prediction.png')
    with inference_lock:
        result = run_prediction(run_id, checkpoint, s.safe_path(im['image_path']), instruction.strip(), threshold)
    return dict(result, image_id=im['id'], image_url=im['url'])


@app.post('/api/predictions/correct')
def correct_prediction(payload: CorrectionPayload):
    im = s.get('image', payload.image_id)
    element_id = s.uid() if payload.target_present else None
    elements, examples = list(im['elements']), list(im['examples'])
    if element_id:
        elements.append({'id': element_id, 'class_id': payload.class_id, 'label': '', 'bbox': payload.bbox, 'click_point': payload.click_point})
    elif any(v is not None for v in (payload.class_id, payload.bbox, payload.click_point)):
        raise ValueError('Absent correction must have null target fields')
    examples.append({'id': s.uid(), 'instruction': payload.instruction, 'target_present': payload.target_present, 'element_id': element_id, 'status': 'draft', 'ambiguous': False})
    return dataset.save_annotations(im['id'], {'elements': elements, 'examples': examples, 'group': im['group']})


@app.post('/api/evaluate')
def evaluate(payload: EvaluationPayload):
    from .inference import evaluate as run_evaluation
    if payload.split not in ('train', 'val', 'test'):
        raise ValueError('Select train, val or test split')
    with inference_lock:
        result = run_evaluation(**payload.model_dump())
    result.setdefault('id', s.uid())
    result.setdefault('created_at', s.now())
    return s.put('evaluation', result['id'], result)


@app.get('/api/evaluations')
def evaluations():
    return s.list_items('evaluation')


@app.get('/api/evaluations/{evaluation_id}')
def get_evaluation(evaluation_id: str):
    return s.get('evaluation', evaluation_id)


@app.get('/api/files/{relative:path}')
def image_file(relative: str):
    path = s.safe_path(relative)
    if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'} or not path.is_file():
        raise HTTPException(404, 'Image not found')
    return FileResponse(path)


@app.get('/api/chat/examples')
def chat_examples():
    return s.list_items('chat_example')


@app.post('/api/chat/examples')
def create_chat_example(payload: ChatExamplePayload):
    from .chat import save_example
    return save_example(payload.model_dump())


@app.put('/api/chat/examples/{example_id}')
def update_chat_example(example_id: str, payload: ChatExamplePayload):
    from .chat import save_example
    return save_example(payload.model_dump(), example_id=example_id)


@app.delete('/api/chat/examples/{example_id}')
def delete_chat_example(example_id: str):
    from .chat import delete_example
    delete_example(example_id)
    return {'deleted': example_id}


@app.get('/api/chat/training/defaults')
def chat_training_defaults():
    from .chat import default_config
    return default_config()


@app.get('/api/chat/runs')
def chat_runs():
    from .chat import recover_runs
    recover_runs()
    return s.list_items('chat_run')


@app.get('/api/chat/sources')
def chat_model_sources():
    from .chat import sources
    return sources()


@app.post('/api/chat/runs')
def create_chat_run(payload: ChatRunPayload):
    from .chat import create_run, launch_run
    run = create_run(payload.model_dump())
    try:
        return launch_run(run['id'])
    except (ValueError, OSError) as exc:
        s.patch('chat_run', run['id'], {'status': 'error', 'error': str(exc), 'pid': None})
        raise ValueError(f'Could not launch chat training: {exc}') from exc


@app.get('/api/chat/runs/{run_id}')
def get_chat_run(run_id: str):
    from .chat import recover_runs
    recover_runs()
    return s.get('chat_run', run_id)


@app.get('/api/chat/runs/{run_id}/download')
def download_chat_model(run_id: str):
    from .chat import export_path
    with inference_lock:
        path = export_path(run_id)
    return FileResponse(path, filename=f'chat-{run_id}.pt', media_type='application/octet-stream')


@app.post('/api/chat/runs/{run_id}/{action}')
def control_chat_run(run_id: str, action: str):
    from .chat import control_run
    if action not in ('stop', 'resume'):
        raise HTTPException(404, 'Unknown chat training action')
    return control_run(run_id, action)


@app.post('/api/chat/predict')
def chat_predict(payload: ChatPredictionPayload):
    from .chat import predict
    with inference_lock:
        return predict(payload.run_id, payload.message)


# Installed frontend assets are served by this same loopback process.
FRONTEND = Path(__file__).resolve().parents[1] / 'frontend' / 'dist'
if FRONTEND.is_dir():
    app.mount('/', StaticFiles(directory=FRONTEND, html=True), name='frontend')
else:
    @app.get('/')
    def setup_hint():
        return {'message': 'Build frontend with npm ci and npm run build inside frontend/, then restart server.', 'api_docs': '/docs'}

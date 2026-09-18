# Setup and operation

Use 64-bit Python 3.13 and Node.js 22.12 or newer (verified with Node.js 24.19.0 and npm 11.17.0). The Windows scripts create `.venv`, install the pinned dependencies, install the frontend lockfile with `npm ci`, build the frontend, and launch one local server. No pretrained assets are downloaded.

## Scripted installation

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cuda
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

`-Device cpu` selects the CPU PyTorch index. The default CUDA build targets CUDA 12.8; driver support and available device memory still determine whether your machine can run a configuration. Use NVIDIA's installed driver information and PyTorch's actual device report when troubleshooting. Do not infer GPU availability from the installation succeeding.

```powershell
.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

The configured laptop has an NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB VRAM. Begin with the conservative default configuration and measure the actual workload. The worker performs a full forward/backward/optimizer memory probe before a CUDA run. If allocation fails, reduce batch size, image resolution, or feature width and create a new run with those explicit settings. The original 4 GB A500 use case can use the same application with conservative settings, but fit on that untested device is not guaranteed. CPU execution is slower but useful for workflow checks.

## Manual installation

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
Push-Location frontend
npm ci
npm run build
Pop-Location
.venv\Scripts\python.exe -m uvicorn grounding.api:app --host 127.0.0.1 --port 8000
```

For CPU PyTorch replace the index with `https://download.pytorch.org/whl/cpu` and install `requirements-cpu-lock.txt` instead of `requirements-lock.txt`. The installed wheel reports a local version suffix such as `+cu128` or `+cpu`. The shorter `requirements.txt` and `requirements-cpu.txt` declare direct dependencies; use the lock files for a fully pinned installation.

## Frontend development

Run the API in one terminal using the manual startup command. In another:

```powershell
Push-Location frontend
npm run dev -- --host 127.0.0.1
```

Vite proxies `/api` requests to `127.0.0.1:8000`. The production server uses the existing `frontend/dist` build; rebuild after frontend changes. API development reload should not be used while testing worker recovery, because process restarts are a separate event from a training worker restart.

## Persistence and shutdown

By default metadata lives in SQLite and assets live below `data/`. To relocate all runtime storage:

```powershell
$env:GROUNDING_DATA_DIR = 'D:\GroundworkData'
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Pause or stop a training run and wait for its checkpoint status before intentionally shutting down. A sudden worker termination may lose work after the last completed checkpoint, but atomic checkpoint replacement preserves the previous valid checkpoint. On startup, stale runs can be recovered and resumed from that saved boundary.

If port 8000 is occupied, pass `-Port 8001` to `start.ps1`; use the matching address in the browser. The Vite proxy assumes port 8000 unless its configuration is changed. Check the Training error panel and the run's worker log for failures.

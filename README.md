# Groundwork — local GUI grounding

Groundwork is a local workspace for annotating browser screenshots, creating immutable datasets, training a small instruction-conditioned grounding model from random weights, evaluating checkpoints, and predicting one requested element. The application uses React/TypeScript, FastAPI, SQLite, and PyTorch. Screenshots and training data stay on your computer.

This is a task-specific research tool. Training from scratch requires representative reviewed examples. A brief synthetic run verifies the workflow; it does not establish useful accuracy on real websites or unrestricted language understanding. The configured laptop has an Intel Core i5-13420H, 16 GB system RAM, and an NVIDIA GeForce RTX 5050 Laptop GPU with 8 GB VRAM; conservative settings also accommodate testing on smaller devices when their memory probe succeeds.

## Windows setup

Requirements: 64-bit Python 3.13, Node.js 22.12 or newer, and an NVIDIA driver compatible with the selected CUDA wheel for GPU training. CPU execution is supported for debugging and inference. Install dependencies while connected to the internet; running the application does not require a hosted model or external API.

From PowerShell in this repository:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
powershell -ExecutionPolicy Bypass -File scripts/start.ps1
```

Open **http://127.0.0.1:8000**. The API reference is at **http://127.0.0.1:8000/docs**. The start script binds to loopback. Keep this local application on loopback: it has no account system or network authentication.

For CPU-only installation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1 -Device cpu
```

See [setup and troubleshooting](docs/setup.md) for manual commands and development mode. Exact dependency pins live in the requirements files and `frontend/package-lock.json`.

## First workflow

1. Configure element classes before versioning data. Upload screenshots under **Images & annotations**, [capture webpages at different resolutions](docs/capture-extension.md) and import their ZIP, or generate explicitly labeled synthetic examples for a demonstration.
2. Draw visible elements, assign classes, and adjust their candidate click points. Create instructions, associate each instruction with an element, or mark its target absent. Save as drafts until you explicitly review them.
3. Review complete, unambiguous examples. Set website, template-family, or collection-session groups so related screenshots stay together in a split.
4. Under **Dataset versions**, validate and create a snapshot. Review split counts; small collections may not populate every split. Dataset snapshots copy the images and preserve annotation content.
5. Under **Training**, select **Fresh model**, choose the dataset and settings, and start training. The worker runs separately from the web server. Use **Pause**, **Resume**, or **Stop** to control it.
6. Select a checkpoint under **Evaluation** to inspect aggregate metrics and individual overlays. Under **Prediction**, upload a screenshot and enter an instruction. Download the result JSON or save a correction as a draft for review.
7. To learn from additional data, review the combined old and new examples, create a new version, and choose **Retrain from weights**. This creates a new run and preserves its parent.

**Resume** restores the same experiment from a training checkpoint. **Retrain** starts a new experiment from existing weights with a fresh optimizer and scheduler. **Fresh** starts all learned weights randomly. An inference export can initialize a retraining run; it cannot exactly resume a training run.

Each screenshot card under **Images & annotations** has a **Delete image** action. Confirming removes the live image and its annotations; existing dataset snapshots, checkpoints, and training runs are preserved.

## Capture webpages at different resolutions

Load the **`capture-extension/`** folder as an unpacked extension in Chrome or Edge. Open a webpage, click **Groundwork Dataset Capture**, select viewport sizes, and click **Capture selected sizes**. The page renders at each size and produces matching PNG screenshots. Export the capture ZIP, then choose **Images & annotations → Import capture ZIP** in the lab to start labeling it for training.

Presets cover desktop, tablet, and phone widths; you can also add custom sizes. Captures stay local, preserve their dataset group, and start unannotated. See the [installation and capture guide](docs/capture-extension.md).

## Teach your own chat replies

Open **Training → Chat** to teach your existing grounding model to reply using examples you write yourself:

1. Enter a user message such as `hey` and the reply you want, such as `Hey! How can I help you?`, then save the example. Add different messages and their desired replies. You can edit or delete examples before training.
2. Select **Grounding model to train for chat**. Choose a saved grounding checkpoint, or a version you have already taught chat. Train a grounding model first if this list is empty.
3. Give the experiment a name, choose the training settings, and click **Train chat model**. Each run saves its own copy of the source weights and your examples. The monitor shows measured loss and progress, with controls to stop and resume.
4. Test its replies in the chat area, or choose the same version under **Prediction** to locate screenshot elements. **Download grounding + chat model** saves both abilities in one `.pt` file, also usable by the browser CLI for grounding.
5. Add or correct examples and select your latest combined model as the source for another teaching run.

The same `Grounder` now supports screenshot grounding and chat. Chat prompts reuse its existing text encoder, with a character embedding and reply decoder trained on your examples. Original grounding weights stay fixed during chat training, preserving its screenshot predictions. Training creates a new model version and keeps the parent checkpoint. Everything runs locally; each chat message is independent, and useful replies depend on the examples you teach.

## Use a trained model in your browser

The local browser CLI loads your generated `.pt` file and runs screenshot-grounded commands or JSON task sequences. It lets you select a running Chrome/Edge browser and one of its open websites through the included extension, or attach to a local debugging endpoint. No external LLM or API key is needed.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/agent.ps1 --model "data\exports\YOUR_EXPORT_ID.pt"
```

Follow the [browser CLI guide](docs/browser-agent.md) to connect your browser, choose a tab, and use `find`, `click`, `type`, `press`, and `run`. The model locates elements; you supply the command sequence. The guide includes a local practice page, a sample task, and an isolated browser integration smoke check.

## Verification

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\smoke.py
.venv\Scripts\python.exe scripts\smoke.py --device cpu
```

The smoke script uses its own temporary data directory. It generates synthetic data, validates and versions it, runs a real training process, pauses and restarts it, checks inference, exports weights, and starts a separate retraining run. It checks parent preservation. See [verification evidence](docs/verification.md) for the actual checks executed during implementation.

Frontend checks run with `npm test` inside `frontend/`. The optional `node tests/ui_smoke.mjs` browser runner verifies the annotation-to-training-to-prediction workflow against a real GPU worker; see the verification report for its one-time Chromium installation step.

Run `node tests/ui_chat.mjs` for the combined workflow. It uses isolated storage and a browser, trains a grounding model, teaches that model two greeting replies, stops and resumes a real CPU worker, checks both chat and grounding predictions, downloads the combined model, and verifies the mobile layout.

Run `node tests/extension_capture.mjs` to test the installed capture extension in an isolated Chromium profile, including responsive PNG capture, page restoration, ZIP export, and lab import. The [capture guide](docs/capture-extension.md#developer-verification) lists setup and focused checks.

## Project and storage

| Path | Purpose |
| --- | --- |
| `frontend/` | React workspace and coordinate helpers |
| `grounding/` | API, persistence, versioning, model, training, evaluation, inference |
| `tests/` | Meaningful model, data, checkpoint, and API checks |
| `scripts/` | Windows setup/start and isolated workflow smoke check |
| `browser-extension/` | Local bridge for selecting and controlling existing Chrome/Edge tabs |
| `capture-extension/` | Standalone extension for collecting webpage PNGs at multiple viewport sizes |
| `examples/` | Local browser practice page and repeatable CLI task |
| `data/` | Default runtime SQLite database, uploads, versions, runs, and exports |

Set `GROUNDING_DATA_DIR` to an absolute folder to choose another runtime location before starting the server. Back up that folder while training and the server are stopped. Existing snapshots do not change when you edit live annotations. Do not manually modify snapshot or checkpoint files.

Further details: [architecture and metrics](docs/architecture.md), [JSONL import/export](docs/dataset-format.md), [resume and retraining](docs/training.md), and [supported behavior and limits](docs/limitations.md).

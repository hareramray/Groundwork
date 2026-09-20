# Verification report

Verified locally on 2026-09-19. These are software acceptance checks using synthetic images, not evidence of accuracy on real websites.

## Unified grounding and chat training (2026-09-20)

- The unified chat suite passed all 29 tests. The combined Python run passed 246 tests; one existing browser-runtime test observed a changing browser viewport height and passed when rerun alone (1 passed in 3.47 seconds). No browser-runtime code was changed.
- CPU and CUDA greeting checks: after 200 chat epochs on a real grounding checkpoint, the same model reproduced `hey` → `Hello!` and `bye` → `Goodbye!`. CUDA selected the RTX 5050, completed in 11.33 seconds, and reached training loss 0.00299. This checks learning on supplied examples, not general conversation quality.
- Every original grounding tensor and the parent checkpoint hash remained unchanged. Grounding predictions before and after chat teaching matched exactly. Hooks verified both tasks use the same text encoder. The combined export loaded in one `FileGrounder` instance for both `predict` and `chat`, and a further grounding retraining run retained the learned replies.
- Other checks cover example/API validation, required model sources, immutable teaching snapshots, character-vocabulary expansion when continuing a combined model, exact CPU model/optimizer recovery after stop/resume, stale workers, and stopping before any learning update.
- Production TypeScript/Vite build and all five frontend unit tests passed.
- Windows concurrent checkpoint reads and replacements reproduced a sharing violation. Bounded retries fixed it; a stress check completed 100 atomic writes alongside 508 validated reads. Each model download now receives its own immutable export file.
- `node tests/ui_chat.mjs` checks the real API and CPU worker: train a grounding source, select it for chat, save examples, stop/resume, generate taught replies, preserve grounding predictions, download both abilities in one file, retain saved data after reload, and render at 1440- and 390-pixel widths. The completed browser run had no horizontal overflow or browser errors, and its screenshots were visually reviewed.

## Environment

- Windows laptop, Intel Core i5-13420H, 8 cores / 12 logical processors, 15.7 GiB usable system RAM.
- NVIDIA GeForce RTX 5050 Laptop GPU, 8,151 MiB reported device memory.
- Python 3.13.15; PyTorch 2.11.0+cu128; CUDA build 12.8.
- Node.js 24.19.0 and npm 11.17.0.
- Pinned direct and transitive Python requirements; included npm lockfile. `pip check` passed.

Per the user's updated instruction, training verification uses this laptop. The originally mentioned A500 with 4 GB VRAM is not available here, so fit on that device is unverified.

## Completed workflow checks

| Check | Observed result |
| --- | --- |
| Data/API acceptance suite | 37 passed in 11.56 seconds |
| Model/training acceptance suite | 14 passed in 21.07 seconds; includes deterministic uninterrupted-versus-resumed comparison across real processes |
| CUDA + mixed precision subprocess smoke | Passed in 42.03 seconds; paused at optimizer step 2, restarted the API and training process, resumed to step 54 |
| CPU subprocess smoke | Passed in 34.59 seconds with retained artifacts; default temporary-directory run passed in 27.67 seconds and cleaned up |
| Prediction | Produced finite presence score, valid normalized box, correctly scaled original-pixel coordinates; threshold 1 abstained with public box/click set to null |
| Prediction correction | Saved a draft requiring explicit review |
| Evaluation | Returned computed metrics, baselines, and per-example results from a real saved checkpoint |
| Inference export | Downloaded a real model artifact; initialized and completed a new retraining run from it |
| Retraining on added data | Created a new version combining old and new synthetic data and a separate child run; parent checkpoint SHA-256 and run metadata stayed unchanged |
| Windows startup scripts | PowerShell parser accepted setup and start scripts |
| Combined Python acceptance suite | 51 passed in 37.08 seconds; no failed assertions |
| Frontend unit checks | 5 passed: coordinates, inverse transformations, editing constraints and validation |
| Production frontend | TypeScript checking and Vite production build passed; npm audit reported 0 vulnerabilities |
| Real browser workflow | Passed in isolated headless Chromium against the actual local API and GPU worker; no browser page errors |
| Responsive layout | 390-pixel mobile dashboard/annotation and 1440-pixel desktop evaluation had no horizontal page overflow; annotation locks background scrolling |
| Image deletion UI | Production build and `node tests/ui_delete.mjs` passed: cancel/Escape, annotation opening, failed-request recovery, duplicate-submit prevention, deletion/count refresh, and preserved snapshot export |

The smoke runs created 18 synthetic screenshots for the first version and added 6 for the retraining version. The small configuration was 64 pixels, batch size 2, gradient accumulation 2, CNN width 8, text dimension 16, three initial epochs, and one epoch for each child. Each stage used the actual HTTP API, server subprocess, persisted SQLite data, and training subprocess. Both smoke runs covered fresh training, pause, application restart, resume, prediction, abstention, evaluation, export, checkpoint-based retraining, and export-based retraining.

Retained local evidence is under `.verification/cuda-smoke-fixed/`, `.verification/cpu-smoke-fixed/`, and `.verification/laptop-default-probe/`. These generated artifacts are intentionally excluded from version control; use `scripts/smoke.py --data-dir <new-empty-folder>` to retain a fresh report and checkpoints.

## Measured GPU allocation

| Configuration | Full-update peak PyTorch allocated memory | PyTorch reserved memory |
| --- | --- | --- |
| Smoke: 64 pixels, batch 2, width 8, accumulation 2, mixed precision | 73.678 MiB | 86 MiB |
| Default: 256 pixels, batch 4, width 32, text dimension 64, float32 | 119.113 MiB | 142 MiB |
| Default: same settings, mixed precision | 112.346 MiB | 142 MiB |

Each probe performed forward, backward, and Adam optimizer-state allocation. The default configurations also completed a real training update after restoring the pre-probe state. These are PyTorch allocation measurements for the tested workload, excluding CUDA context and other applications; they are not total device-memory use or guarantees for other settings.

Verification exposed and fixed mixed-precision overflow handling and Windows 64-bit process-memory reporting. AMP overflow retries reuse the same batch and RNG boundary without advancing the optimizer-step count, scheduler, or data cursor until an update succeeds.

## Acceptance coverage

Data/API tests cover normalized annotation editing, explicit review and exclusion, target deletion integrity, absent-target nulls, rejected invalid saves preserving prior annotations, synthetic seed determinism, grouping and identical-image split leakage, immutable snapshots, image/metadata tampering, JSONL/image roundtrip preservation, malformed and oversized import rejection before writes, Windows case-distinct and reserved image IDs, class mapping protection, draft corrections, SQLite persistence, and allowed local origins/ports. The data/API run emitted two third-party Starlette deprecation warnings; assertions passed.

Model/checkpoint tests cover bounded boxes, masked losses, all-absent batches, shared preprocessing, tokenizer scope, checkpoint atomicity, compatibility checks, deterministic process resumption, retraining lineage, and prediction abstention. The controlled CPU restart comparison checks exact model and optimizer/scheduler state equality against uninterrupted training; it does not claim bit-identical results across arbitrary hardware or dependency changes.

The browser acceptance runner (`tests/ui_smoke.mjs`) uploaded a screenshot, drew and labeled a target, associated an instruction, explicitly reviewed and saved it, then moved/resized it under zoom, adjusted its click point, and deleted a second box. It compared saved normalized coordinates against expected values and confirmed edits require review again. It also created a dataset version excluding drafts, started/paused/resumed a real GPU run, downloaded an inference export, created a separate retraining child with inherited architecture, ran prediction, checked stale results disappear after instruction changes, saved a correction as draft, and rendered actual evaluation metrics and overlays. Desktop and narrower annotation screenshots were visually inspected. Final browser artifacts: `test-results/ui-1789769030707/` (ignored generated data).

The combined Python run emitted the two Starlette deprecation warnings and an optional globally installed Triton CUDA-toolkit discovery warning. The PyTorch CUDA runtime, GPU probes, and GPU workflows passed; no standalone CUDA compiler toolkit is needed by this application.

Run the current checks yourself:

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe scripts\smoke.py --device cuda
.venv\Scripts\python.exe scripts\smoke.py --device cpu
Push-Location frontend
npm test
npm run build
Pop-Location
```

To run the full browser check (a test-only Chromium download is required once):

```powershell
Push-Location frontend
npx playwright install chromium
Pop-Location
node tests/ui_smoke.mjs
```

The browser runner creates an isolated local dataset and retains screenshots/results under `test-results/`. It does not operate your normal browser profile or use your real screenshots.

## Local browser CLI verification

The new browser agent was checked against a disposable headless Chromium instance and the bundled loopback practice page, using Python Playwright 1.63.0. It did not operate existing user browser tabs.

| Check | Result |
| --- | --- |
| Full Python regression suite | 210 passed in 37 seconds; final model-menu fix additionally passed all 26 CLI tests |
| CDP integration with CPU inference | Passed: real saved `.pt` loading/inference, browser and tab selection, typing, Enter, model-grounded clicks, abstention, scrolling, and navigation |
| CDP integration with this laptop's CUDA device | Passed the same workflow, including the actual CLI subprocess executing the JSON task |
| High-DPI coordinates | Passed at device scale factor 2; screenshot dimensions match the CSS viewport scale, including pages with scrollbars |
| Capture after scrolling | Fixed-position target pixels remained aligned after scrolling, and a fresh grounded click succeeded |
| Loaded browser extension | Passed with the actual MV3 extension: popup token connection, running browser/tab listing and selection, screenshot inference, typing, Control+A/Backspace, Enter, click, abstention, scroll, and navigation |
| Selected-tab boundaries and exit | Selected and unrelated tabs remained open after CDP detach and extension disconnect; unrelated content was unchanged |
| Local reports | Before/after PNGs and JSONL action reports were produced; action text fields were omitted from the logs |
| PowerShell entry point | `scripts/agent.ps1 --help` forwarded arguments and exited successfully |

The `.pt` used by these integration checks is an explicitly labeled **constant-output fixture**, built with the real Groundwork architecture and export schema. Its center prediction is intentionally fixed. These checks establish that the saved-model-to-browser pipeline works; they do not establish learned accuracy or general task planning ability. Model quality must still be evaluated with independent reviewed screenshots from the intended websites. Page titles, URLs, or screenshots in real reports may contain entered text even though the action's `text` field is redacted.

The high-DPI check exposed a scrollbar-width mismatch between the captured PNG and the CSS viewport. Both transports now clip screenshots to the exact reported viewport before converting normalized model coordinates to browser input coordinates.

Retained reports from the passing checks are under `data/agent-smoke-verification/`, `data/agent-smoke-verification-cuda/`, and `data/agent-extension-smoke-verification/` (ignored runtime artifacts). Reproduce them with:

```powershell
.venv\Scripts\python.exe -m playwright install chromium
.venv\Scripts\python.exe scripts/agent_smoke.py --output-dir data/agent-smoke-verification
.venv\Scripts\python.exe scripts/agent_smoke.py --device cuda --output-dir data/agent-smoke-verification-cuda
.venv\Scripts\python.exe scripts/agent_smoke.py --extension --output-dir data/agent-extension-smoke-verification
```

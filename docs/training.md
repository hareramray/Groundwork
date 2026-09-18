# Training, resume, and retraining

| Action | Learned parameters | Optimizer and scheduler | Dataset | Identity |
| --- | --- | --- | --- | --- |
| Fresh model | Random initialization | New | Selected immutable version | New run |
| Resume | Latest compatible training checkpoint | Restored | Original immutable version | Same run |
| Retrain from weights | Selected parent checkpoint/export | New | New compatible immutable version allowed | New run with parent lineage |

Training runs in a separate Python worker process, using the same interpreter and runtime directory as the API. Run settings are persisted before launch. The UI polls persistent progress, while pause and stop requests are consumed by the worker at safe training boundaries.

## Checkpoint boundaries

Checkpoints are saved at optimizer-step boundaries after all microbatches in that gradient-accumulation group have completed. They are also saved at initialization, epoch completion, and pause/stop. A checkpoint records the exact next-example cursor and epoch. Each epoch order is recreated by `torch.randperm` with seed `seed+epoch*104729`, so resumption continues at that saved data-progress boundary. Gradients from a partially executed accumulation group are not checkpointed. Sudden termination can replay up to `checkpoint_every-1` completed updates after the last successful checkpoint, plus unfinished work. A completed checkpoint is written to a temporary file and atomically replaced into its final path.

The training checkpoint contains model weights, optimizer and scheduler state, optional mixed-precision scaler state, epoch and optimizer step, random-generator states, tokenizer vocabulary, class mapping, architecture and preprocessing settings, snapshot identity and fingerprint, data progress, and the best validation metric. Resume validates compatibility before loading, including the actual device, PyTorch version, and mixed-precision execution mode. Changing settings, classes, architecture, source data, or snapshot content does not create an implicit resume. If the original execution environment is unavailable, create a retraining run from its weights instead.

Pause and stop both save a resumable checkpoint when the worker reaches its next safe boundary. They are distinct recorded statuses; resume is available from either. After an unclean process interruption, resume starts from the latest valid saved checkpoint. It cannot reconstruct unsaved work.

Exact CPU replay is checked under controlled seeds and deterministic execution. Floating-point behavior, CUDA kernels, mixed precision, driver updates, different hardware, and different PyTorch versions can change numerical results. Preserve the dependency environment and use CPU for the deterministic acceptance check.

## Additional data

Add and review new images, then create a new immutable version containing the previous and new reviewed examples. Choose **Retrain from weights** and select a source checkpoint. This preserves the parent run and starts a new optimizer/scheduler. Retaining earlier examples can reduce forgetting but does not guarantee it.

The initial implementation freezes the parent's tokenizer and class mapping. New words map to the existing unknown-token entry; it does not expand embeddings or class heads. Inspect unknown-word coverage and maintain compatible class meanings. For a substantially changed instruction vocabulary or class schema, train a fresh model. An inference export provides weights and configuration for retraining, but lacks the optimizer/data-progress state required for exact resume.

## Memory and configuration

Use conservative defaults for a 4 GB GPU and inspect the measured probe and run memory. A full-update probe includes forward pass, backward pass, and optimizer-state allocation. It cannot guarantee that all future workloads will fit. Mixed precision is enabled only on supported devices; CPU debugging uses full precision. Out-of-memory errors retain the most recent valid checkpoint and include actionable reductions in batch size, image resolution, or feature width.

Configuration changes belong to a new run. Resuming an unchanged checkpoint is distinct from creating a new run with smaller settings or retraining from its weights. Gradient accumulation increases effective batch size without requiring every sample to reside on the GPU simultaneously; it still requires enough memory for one microbatch.

## Checkpoint trust

Training checkpoints include Python/PyTorch state and are application-generated local artifacts. Load only checkpoints you trust. Dataset import accepts image/JSON records, not arbitrary executable training checkpoints. Back up the whole runtime directory to preserve both files and SQLite run metadata.

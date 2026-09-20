# Supported behavior and limitations

- Screenshot grounding predicts one requested visible element per instruction. That model does not discover every element or execute browser actions.
- Chat training adds a reply branch to the same grounding model and reuses its text encoder. Original grounding weights are frozen during chat training. It learns single-message replies from your examples without conversation memory. Training loss measures fit to your examples, not accuracy on unseen conversations; add examples and test different messages to assess its behavior.
- No pretrained weights or OCR are used. Small from-scratch models need enough representative examples and may struggle with text reading, unseen wording, tiny controls, or unfamiliar page layouts.
- The target-presence score is an uncalibrated model score. A high score does not prove the box is correct or a click will be safe or successful.
- Predicted click points are box centers. Human annotation points are stored and validated, but the initial model does not learn a separate click-point head.
- The vocabulary is learned from the training split. Retraining retains the original vocabulary and class mapping; new terms become unknown tokens. Vocabulary or class expansion is not implemented.
- Review is explicit. Synthetic examples and prediction corrections enter as drafts unless explicitly reviewed. Synthetic data verifies software behavior and offers no real-world accuracy claim.
- Group metadata is user supplied. The application cannot automatically identify every website/template relationship that could cause train/test leakage.
- CPU mode is supported and can be slow. The GPU probe measures a real update, but a successful probe does not guarantee every future allocation fits. Laptop verification uses the RTX 5050 with 8 GB VRAM; the originally mentioned 4 GB A500 is untested.
- Persistence relies on the local runtime directory. Snapshot integrity is checked; deleting local files outside the application can make records or checkpoints unavailable.
- Training is designed for one local operator and permits one active training worker. Heavy inference alongside training still competes for the same GPU memory.
- The application binds to loopback and has no multi-user authentication. It is not intended as an internet-facing service.
- Abrupt interruption can lose work after the last checkpoint. Exact numerical reproducibility is scoped to controlled deterministic CPU tests, not arbitrary GPUs, drivers, or library versions.

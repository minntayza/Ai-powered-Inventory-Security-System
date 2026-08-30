# AI-Powered Inventory Security System

A local exhibition MVP that combines YOLO inventory/person detection, DeepFace
authorization, stable inventory counting, a deterministic theft state machine,
SQLite audit logging, local/Telegram alerts, a Streamlit dashboard, and optional
Florence-2 visual questions.

## Team installation

Install 64-bit Python 3.11 on every laptop and clone/copy the same repository.
Then run one command from the repository root in PowerShell:

```powershell
.\scripts\setup_windows.ps1
```

The setup script creates `.venv`, detects an NVIDIA GPU, installs the tested CUDA
12.8 PyTorch build when available, otherwise installs CPU-only PyTorch, installs
the same pinned application versions, and verifies imports/configuration. Override
automatic detection when needed:

```powershell
.\scripts\setup_windows.ps1 -Device CUDA
.\scripts\setup_windows.ps1 -Device CPU
```

If PowerShell blocks local scripts for the current session, use:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
```

Do not copy `.venv` between laptops. Each member should generate it locally because
CPU and CUDA wheels differ. Runtime configuration and application code remain the
same on every laptop because `device: auto` selects the available hardware.

### Apple M2 MacBook

On an Apple Silicon Mac running macOS 14 or newer, install Python 3.11 and run:

```bash
bash scripts/setup_macos.sh
```

The script installs native ARM64 PyTorch wheels and verifies the `mps` Metal GPU
backend. The same `device: auto` configuration selects `mps`; unsupported MPS
operations and model-level accelerator failures fall back to CPU. Start the app with:

```bash
.venv/bin/python -m streamlit run src/module_c_ui_dashboard/app.py
```

Configuration lives in `configs/`. The default setup uses laptop webcam index `0`
and pretrained `yolov8n.pt`. It tracks people and counts classroom-relevant COCO
objects such as bags, drink containers, chairs, electronics, books, clocks, and
scissors. Telegram event routing is enabled in the default configuration; set the
two credentials shown in `.env.example` before starting the dashboard.

Telegram notifications are configured for both authorized removals and suspected
theft. Copy `.env.example` to `.env`, replace both placeholder values, and then
start the dashboard. The application loads `.env` automatically; variables already
set by the launching process take precedence. Authorized notifications include the
removed inventory and recognized actor but do not activate the siren.

```powershell
Copy-Item .env.example .env
notepad .env
```

Place 5–10 clear enrollment photos per authorized user under:

```text
assets/known_faces/<person-name>/photo.jpg
```

Add a siren MP3 at `assets/audio/siren.mp3`, or disable the siren in configuration.

Authorized users can also be enrolled from the dashboard with 1-5 clear JPG or
PNG images; three images are recommended. New face images are ignored by Git, but
this repository is inside OneDrive, so review synchronization/privacy settings.

## Run

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/module_c_ui_dashboard/app.py
```

The first YOLO/DeepFace/Florence/Whisper execution may download model weights.
Florence-2 loads only after the first visual question and the local Whisper tiny
English model loads only after the first voice recording, so monitoring can run
without either feature.
Florence uses its supported detailed-caption task for scene/action questions and
its OCR task for questions about visible text; it is not treated as a chat model.

The dashboard always publishes the newest raw webcam frame as the smooth live
preview, including while models initialize. AI boxes, counts, and decisions update
separately at the processed-frame rate, and face recognition runs on a background
worker. Check whether YOLO can use the NVIDIA GPU with:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If this prints `False`, YOLO and Florence run on the CPU and will be slower. Install
a Windows CUDA-enabled PyTorch build appropriate for the installed NVIDIA driver
before enabling the VLM for a live demonstration.

### Classroom monitoring workflow

1. Wait for the camera and models to become ready.
2. Configure the **Monitored zone** tightly around the demonstration desk or shelf.
3. Place protected objects in the zone, stop handling them, and step away.
4. Wait for the stable protected count, then click **Set this count as baseline and arm**.
5. A stable removal is classified exactly once as authorized, suspected theft, or
   unattributed. Monitoring then pauses automatically for that baseline discrepancy.
6. If the item is returned, click **Item returned — resume original baseline**. If
   the inventory change is intentional, click **Use current count as the new baseline**.
   Use **Clear baseline and start over** whenever the scene becomes confusing.

The default live-demo timing uses a seven-frame inventory window, two matching
confirmation frames, and a three-second security grace period. At 3–5 processed
FPS, a suspected-theft decision normally appears about 4–5 seconds after the item
leaves the monitored zone. Authorized removals finish as soon as the count settles.

Use a desk/shelf-sized monitored zone for pickup detection. With **Full frame**
selected, an item carried by a person remains part of the detected inventory until
it leaves the camera view. The dashboard now shows possible removals while they are
being stabilized and records both authorized and unknown-person removals live. Do
not hold an item while establishing the baseline: place it in the zone first and
wait for the count to settle.

Protected classes are defined under `inventory_policy.protected` in
`configs/model_config.yaml`. Contextual classes such as chairs, TVs, and clocks
remain visible but never trigger theft. Uploaded videos run in simulation mode,
which forcibly disables Telegram and the siren.

Confirmed live incidents retain an image and bounded pre/post-event video when
OpenCV encoding is available. Default media retention is 30 days. See
`docs/CLASSROOM_DEMO_FEATURE_IMPLEMENTATION_PLAN.md` for the architecture,
contracts, test matrix, and complete demonstration workflow.

While the dashboard is running, each confirmed incident also schedules a
Florence-2 report in the background. Its pending/completed/failed status and
human-readable result are stored with the SQLite event and shown in the alert and
recent-activity data. The report is descriptive only and never changes the
deterministic security decision. The first report may take longer while Florence
loads or downloads its model weights.

Hardware preparation is documented under `hardware_setup/`. Automated evidence
and the still-required physical exhibition checks are tracked in
`docs/QA_VALIDATION_RESULTS.md`.

Run the engine without the dashboard with:

```powershell
python -m src.backend_alerting.event_loop
```

## Custom inventory classes

The repository includes `docs/CUSTOM_YOLO_TRAINING.md`, a dataset YAML, and a
training script. Build the complete pipeline with pretrained weights first, then
collect and label real webcam scenes. Include `person` in the custom dataset if the
same trained checkpoint will provide person tracking. After training, update the
YOLO model path and target class names in `configs/model_config.yaml`.

## Tests

```powershell
python -m pytest -q
```

Runtime data is stored under `data/logs/` and excluded from Git. VLM answers are
descriptive only; they never authorize people or change theft decisions.

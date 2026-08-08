# Training custom inventory classes with YOLO

This project uses Ultralytics YOLO detection. A custom model needs labeled bounding
boxes, a dataset YAML file, training, validation, and then its `best.pt` checkpoint
passed to `YOLODetector`.

## 1. Choose the classes

Edit `configs/inventory_dataset.yaml`. Keep `person` if the same model will feed the
person/face security pipeline. Add product classes in a stable, zero-based order:

```yaml
names:
  0: person
  1: water_bottle
  2: cereal_box
```

Do not reorder the names after labeling. Similar products should be separate classes
only when the camera images make them visually distinguishable.

## 2. Capture and label images

Collect images from the real camera angle with the lighting, shelf layouts,
occlusion, empty shelves, and people expected in production. As a starting point,
aim for at least 100 varied labeled instances per class; harder classes need more.

Use a bounding-box annotation tool and export in YOLO format. Each image has a
matching text file. Each row is:

```text
class_id x_center y_center width height
```

The four coordinates are normalized from 0 to 1. For example:

```text
1 0.512 0.438 0.120 0.310
```

Create this layout:

```text
data/inventory_yolo/
  images/train/    labels/train/
  images/val/      labels/val/
  images/test/     labels/test/
```

Split by recording session or camera, not random neighboring video frames. A useful
starting split is 70% train, 20% validation, and 10% test. Include negative images
(no target object) with an empty label file. Do not put the same scene in two splits.

## 3. Check labels before training

Make sure every image has a matching label file, class IDs exist in the YAML, boxes
are tight, and no target objects are left unlabeled. Label quality usually matters
more than increasing the epoch count.

## 4. Train

From the repository root, activate the virtual environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/train_yolo.py --data configs/inventory_dataset.yaml --model yolov8n.pt --epochs 50 --device 0
```

Use `--device cpu` without a supported GPU (training will be slower). If GPU memory
is exhausted, reduce `--batch` (for example, `--batch 8`). Resume an interrupted run
with Ultralytics directly:

```powershell
yolo detect train resume model=runs/inventory/train/weights/last.pt
```

## 5. Evaluate

Inspect `runs/inventory/train/` for precision, recall, mAP, confusion matrix, and
validation predictions. Also test unseen full camera recordings. Inventory counting
needs stable recall across frames; a good overall mAP can still hide one weak class.

Common fixes:

- missed objects: add small, occluded, dim, and distant examples;
- false positives: add those backgrounds as negative examples;
- confused products: correct labels and add views emphasizing their differences;
- good validation but poor camera results: remove split leakage and collect more
  production-camera data.

## 6. Use the trained model

The best checkpoint is normally:

```text
runs/inventory/train/weights/best.pt
```

Load it with the exact inventory class names from the YAML:

```python
from src.module_a_perception_engine.yolo_detector import YOLODetector

detector = YOLODetector(
    model_path="runs/inventory/train/weights/best.pt",
    inventory_classes={"water_bottle", "cereal_box"},
)
```

To treat every non-person class as inventory, pass `inventory_classes=None`. The
detector resolves names from the model, so custom numeric class IDs are supported.

## 7. Tune deployment

Start with confidence `0.5`, then tune it using unseen recordings. Version the dataset
YAML and training arguments with each released checkpoint. Do not commit large raw
datasets or generated `runs/` output to Git.

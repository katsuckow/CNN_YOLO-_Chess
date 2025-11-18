
# Chess Piece Detection — CNN & YOLO (Ultralytics)

This repository contains code and notes for detecting chess pieces using two complementary approaches:
- a small Convolutional Neural Network (CNN) for single-piece classification
- Ultralytics / YOLO for object detection on full-board images

Source documentation and analysis are in [CNN_Ultralytics_Yolo_Chess.Rmd](CNN_Ultralytics_Yolo_Chess.Rmd) and the exported [CNN_Ultralytics_Yolo_Chess.html](CNN_Ultralytics_Yolo_Chess.html).

## Project layout
- [CNN/CNN_ChessPieces.py](CNN/CNN_ChessPieces.py) — main CNN training script (see symbols [`load_images`](CNN/CNN_ChessPieces.py) and [`model`](CNN/CNN_ChessPieces.py)).
- [import_jpg.py](import_jpg.py) — simple image loader (function: [`load_images`](import_jpg.py)).
- [y.txt](y.txt) — numeric labels per image.
- [z.txt](z.txt) — class names.
- [CNN/README_CNN.md](CNN/README_CNN.md) — CNN script notes and recommendations.
- [CNN_Ultralytics_Yolo_Chess.Rmd](CNN_Ultralytics_Yolo_Chess.Rmd) — analysis & Ultralytics examples.
- [CNN_Ultralytics_Yolo_Chess.html](CNN_Ultralytics_Yolo_Chess.html) — rendered report.

## Quickstart — environment
Install required packages:
```sh
pip install tensorflow numpy matplotlib pillow scikit-learn ultralytics opencv-python
```

## Quickstart — run CNN classifier
1. Put single-piece images in `pics/` (default used by the scripts).
2. Ensure labels in [y.txt](y.txt) match the order of images loaded. The loader in [import_jpg.py](import_jpg.py) and [CNN/CNN_ChessPieces.py](CNN/CNN_ChessPieces.py) uses `os.listdir()` — consider using the sorted variant (`sorted(os.listdir(...))`) as suggested in [CNN/README_CNN.md](CNN/README_CNN.md).
3. Run training:
```sh
python CNN/CNN_ChessPieces.py
```
Notes:
- The CNN in [`CNN/CNN_ChessPieces.py`](CNN/CNN_ChessPieces.py) defines `layers.Dense(12, activation='softmax')` — change to `num_classes = len(np.unique(y))` and update the final Dense layer if your dataset differs (see [CNN/README_CNN.md](CNN/README_CNN.md)).
- Outputs: training/validation plots, confusion matrix, and in-script printed predictions.

## Quickstart — YOLO (Ultralytics)
See the analysis and example commands in [CNN_Ultralytics_Yolo_Chess.Rmd](CNN_Ultralytics_Yolo_Chess.Rmd) and the rendered [CNN_Ultralytics_Yolo_Chess.html](CNN_Ultralytics_Yolo_Chess.html). Typical training example:
```py
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
model.train(data='ChessPieces.yaml', epochs=10, imgsz=256, batch=10, augment=True)
```
Prepare YOLO dataset layout (`images/train`, `images/val`, `labels/train`, `labels/val`) and a `ChessPieces.yaml` that lists class names and paths.

## Recommendations & troubleshooting
- Label alignment: ensure the order of [y.txt](y.txt) matches images read by [`load_images`](CNN/CNN_ChessPieces.py) or [`load_images`](import_jpg.py). Use sorted filenames or numeric prefixes.
- Image channels: verify all images are RGB; convert grayscale to RGB or filter them out to avoid shape errors.
- Small dataset: increase data with augmentation (see [CNN/README_CNN.md](CNN/README_CNN.md)) or use a pretrained backbone.
- Save model: add `model.save('model.h5')` in [`CNN/CNN_ChessPieces.py`](CNN/CNN_ChessPieces.py) after training.
- If using Ultralytics, run on GPU-enabled environment for reasonable training times.

## References
- CNN script: [CNN/CNN_ChessPieces.py](CNN/CNN_ChessPieces.py) (symbols: [`load_images`](CNN/CNN_ChessPieces.py), [`model`](CNN/CNN_ChessPieces.py))
- Image loader: [import_jpg.py](import_jpg.py) (symbol: [`load_images`](import_jpg.py))
- Labels: [y.txt](y.txt), [z.txt](z.txt)
- Report: [CNN_Ultralytics_Yolo_Chess.Rmd](CNN_Ultralytics_Yolo_Chess.Rmd) and [CNN_Ultralytics_Yolo_Chess.html](CNN_Ultralytics_Yolo_Chess.html)
- Notes: [CNN/README_CNN.md](CNN/README_CNN.md)

## Next steps
- Increase dataset size and diversity.
- Add model checkpointing and a small CLI to control dataset paths and model hyperparameters.
- Optionally switch CNN to a pretrained backbone (ResNet, MobileNet) for transfer learning.


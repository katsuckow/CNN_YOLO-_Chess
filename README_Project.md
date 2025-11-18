# Chess Piece Detection — CNN & YOLO (Ultralytics)

This project explores detecting and locating chess pieces on images using two approaches:
- a simple Convolutional Neural Network (CNN) classifier for single-piece images
- object detection with YOLOv5/Ultralytics to detect multiple pieces in full-board images

The analysis, code examples and experiments are documented in `CNN_Ultralytics_Yolo_Chess.Rmd` and the included notebooks and scripts.

**Project Structure**
- **`CNN/`**: helper scripts and the CNN implementation (`CNN/CNN_ChessPieces.py`).
- **`images/`**: images used for training and validation.
  - `images/train/`, `images/val/`
- **`labels/`**: YOLO-format label files.
  - `labels/train/`, `labels/val/`
- **`ChessPieces.yaml`**: dataset config for Ultralytics/YOLO training.
- **`Ultralytics_YOLO.ipynb`**, `CNN_Ultralytics_Yolo_Chess.Rmd`: notebooks and documentation.

**Dataset**
- Images were labeled using `https://makesense.ai` and converted into the YOLO format.
- Expected directory layout for YOLO/Ultralytics:
  - `images/train/`, `images/val/`
  - `labels/train/`, `labels/val/`
- The YAML file `ChessPieces.yaml` contains paths to the image and label folders, class names and number of classes.

**Quick Setup**
- Recommended: run training on Colab/GPU or a local machine with CUDA-enabled PyTorch.

In PowerShell (Windows):

```powershell
# clone yolov5 (if training locally with the repo)
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

Install core Python packages (some may already be installed by `yolov5` requirements):

```powershell
pip install ultralytics opencv-python matplotlib numpy
# Install PyTorch per your system from https://pytorch.org (CPU or CUDA variant)
```

Or in Google Colab you can run the commands shown in `CNN_Ultralytics_Yolo_Chess.pdf` to mount Google Drive and train directly.

**Training (Ultralytics / YOLO)**
- Ensure `ChessPieces.yaml` points to the correct `images` and `labels` directories and lists the `names` (class labels).
- Example Python snippet (from html and pdf report files) to train with Ultralytics API:

```python
from ultralytics import YOLO
model = YOLO('yolov5s.pt')
model.train(data='ChessPieces.yaml', epochs=10, imgsz=256, batch=10, save_json=True, lr0=0.05, augment=True)
```

- Example prediction:

```python
result = model.predict('images/val/whitecastle2.jpg', save=True, imgsz=256, conf=0.45, iou=0.5)
```

**CNN Approach**
- A simple CNN classifier for single-piece images is implemented in `CNN_ChessPieces.py`.
- Images are resized to fixed shape (example: 224x224 or 512x512) and labels are one-hot encoded for 12 classes.
- The Rmd documents why the CNN approach was insufficient for full-board detection (background variability, etc.).

**Notes & Findings**
- The CNN classifier shows limited performance for full-board detection because of variable backgrounds and perspectives.
- YOLOv5 (Ultralytics) is more suitable for detecting multiple pieces in an image; labeling images and providing diverse perspectives improves results.
- Current limitations: dataset size, noisy/overhead images, and insufficient variation in training images.

**Recommended Next Steps**
- Collect and label more images with diverse perspectives and lighting.
- Exclude overhead/top-down images if they confuse the detector, or include more top-down samples to learn that perspective.
- Implement board/feld detection (find chessboard grid using classical CV or a dedicated model), then map detected piece boxes to board coordinates.

**References & Resources**
- Ultralytics YOLO: https://github.com/ultralytics/yolov5
- Labeling tool used: https://makesense.ai

**Contact / Author**
- Katja Suckow: project files and documentation located in this repository. For more information `CNN_Ultralytics_Yolo_Chess.pdf`.


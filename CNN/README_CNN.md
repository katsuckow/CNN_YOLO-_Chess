# CNN_ChessPieces.py — README

Brief README for the `CNN_ChessPieces.py` training script.

## Purpose

- Train a small Convolutional Neural Network to classify chess piece images.
- Provides simple data loading, preprocessing, a Keras CNN model, training, and evaluation (plots + confusion matrix).

## Requirements

- Python 3.8+
- TensorFlow (tested with TensorFlow 2.x)
- NumPy
- Matplotlib
- Pillow (PIL)
- scikit-learn

Install dependencies with pip:

```bash
pip install tensorflow numpy matplotlib pillow scikit-learn
```

## File / Data layout

- `CNN_ChessPieces.py` — main script.
- A folder named `pics` (relative to the script) containing the images to load (the script currently picks `.jpg` files).
- `y.txt` — a text file with numeric labels corresponding to the images in `pics`.

Important: The script uses `os.listdir(image_folder)` to collect images and then `np.loadtxt('y.txt')` to load labels. `os.listdir` does not guarantee a stable ordering across platforms, so make sure the order of `y.txt` matches the order in which images are read. A safer approach is to sort filenames (or name files with a numeric prefix) so labels align with images.

## How the script works (quick)

- Loads images from `pics` into a NumPy array (keeps original channels).
- Resizes all images to `224x224` using `tf.image.resize`.
- Applies `tf.keras.applications.resnet50.preprocess_input` to inputs.
- Loads labels from `y.txt` and one-hot encodes them using `LabelBinarizer`.
- Splits data into train/test with `train_test_split(test_size=0.2)`.
- Defines a small Sequential CNN in Keras and trains for 20 epochs with `batch_size=2`.
- Produces training/validation plots and a confusion matrix.

## Run

From the folder containing the script run:

```bash
python CNN_ChessPieces.py
```

You can edit the script to change:
- the `image_folder` path passed to `load_images()` (default: `pics`)
- the label file path (default: `y.txt`)
- model parameters such as number of output units in `Dense(12, activation='softmax')` (must match number of unique labels)

## Notes & Recommendations

- Label alignment: ensure `y.txt` corresponds exactly to the order of images loaded. Consider changing `load_images()` to use `sorted(os.listdir(image_folder))` to force deterministic ordering.
- Output units: the final dense layer currently uses `12` classes. If your dataset has a different number of classes, change `layers.Dense(12, ...)` accordingly or compute the number dynamically (e.g., `num_classes = len(np.unique(y))`).
- Normalization: the script uses `x = x / x.max()` after `preprocess_input`. Verify this suits your pipeline; `preprocess_input` often returns values already in the model's expected range.
- Model saving: add `model.save('model.h5')` after training if you want to persist the trained model.
- Data augmentation: for small datasets use `ImageDataGenerator` or `tf.keras.layers.RandomFlip/Rotation/Zoom` to reduce overfitting.
- Batch size / epochs: `batch_size=2` and `epochs=20` are small; increase them when you have more images.

## Example changes (quick snippets)

- Sort filenames in `load_images`:

```python
for filename in sorted(os.listdir(image_folder)):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        ...
```

- Set number of classes automatically:

```python
num_classes = len(np.unique(y))
model.add(layers.Dense(num_classes, activation='softmax'))
```

## Troubleshooting

- If you get shape errors, check that all images have 3 channels (RGB). Convert grayscale images to RGB or filter them out.
- If training is unstable, try reducing the learning rate or increasing batch size.

## Contact

If you want me to extend the script (save model, use ImageDataGenerator, or switch to a pretrained backbone), tell me what you prefer and I can implement it.

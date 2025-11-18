
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import io
import time
import os

from PIL import Image

# Function to load images into a NumPy list
def load_images(image_folder):
    images = []
    
    # define your image folder here
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg')):  # Change extensions based on your needs
            image_path = os.path.join(image_folder, filename)
            
            # Open image and convert to NumPy array
            img = Image.open(image_path)
            img_array = np.array(img)
            
            # Append the image array to the list
            images.append(img_array)
    
    return images

images = load_images("pics")
images = np.array(images)
print(images, images.shape, type(images))

images_resized = tf.image.resize(images, [224, 224])


# Mit Hilfe einer Preprocess-Funktion aufbereiten
inputs = tf.keras.applications.resnet50.preprocess_input(images_resized)



# In x, y aufteilen
x = inputs
y = np.loadtxt("y.txt")

#z = np.loadtxt("z.txt")

print("X format",x[1:10,:,:], x.shape)
print("Y format",len(y))
#print("Z format", z)
print(x.shape)


z = np.unique(y)



x = np.array(x)

x = x/x.max()



# Einige Bilder darstellen
for i in range(12):
    plt.subplot(3, 4, i+1)  # Subplot mit 3 Zeilen und 4 Spalten
    plt.imshow((x[y == i][0]))
    plt.title(z[i], fontsize=8)
    plt.axis('off')  # Keine Achsenbeschriftung
plt.show()


print(type(x),x.shape,y.shape,len(y))

print("amount of labels",np.unique(y), len(np.unique(y)))
unique, counts = np.unique(y, return_counts=True)






X_train, X_test, y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# %% Tensorboard Callback



# One-hot encode the labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)




#y_proba = model.predict(inputs)

# Build the CNN model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),           # <- add this
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(12, activation='softmax'),
])



# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Train the model
# with more pictures in the dataset, epochs and batch size should be increased
history = model.fit(X_train, y_train, epochs=20, batch_size=2, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)

# Export the label for each class
for i in range(10):
    print("y_true:", np.argmax(y_train[i]), "y_pred:", np.argmax(y_pred[i]))
    print()

# export label with likelihood of prediction
for i in range(10):
    print("y_true:", np.argmax(y_train[i]), "y_pred:", np.argmax(y_pred[i]),
          "certainty:", y_pred[i].max())
    print()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# Compute predicted class indices and true class indices
y_pred_probs = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

# y_test is one-hot encoded -> convert to label indices
y_true_labels = np.argmax(y_test, axis=1)

# Confusion matrix and display
cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=(lb.classes_ if hasattr(lb, 'classes_') else None))
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.show()
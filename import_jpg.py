import os
import numpy as np
from PIL import Image

# Function to load images into a NumPy list
def load_images(image_folder):
    images = []
    
    # Iterate through all files in the specified directory
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
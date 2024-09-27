import numpy as np
from PIL import Image
from transformers import pipeline

# Function to load and convert an image to array
def load_and_convert_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    return img_array

# Load pre-trained model for background removal
pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)

# Function to remove background
def remove_bg(image_path):
    pillow_image = pipe(image_path)
    return pillow_image
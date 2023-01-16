from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import numpy as np

# utilizare model antrenat pentru a prezice clasa unei noi imagini
classifier = load_model('model.h5')

img_path = '/Users/dragosflow/UVT/IA/cats_and_dogs_small/test/dogs/dog.1504.jpg'
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = classifier.predict(x)

if preds[0][0] == 1:
    print("câine")
else:
    print("pisică")

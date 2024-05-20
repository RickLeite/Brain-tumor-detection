import tensorflow as tf
from tensorflow import keras
import os
import cv2
import numpy as np
from keras.preprocessing import image

print(tf.version.VERSION)

model = tf.keras.models.load_model('modelo_brain_tumor_classifier.keras') # Load the model
model.summary() # summary of model model - convolutional and dense layers with shape and params

img = cv2.imread('/app/no/No22.jpg')

img = cv2.resize(img, (128, 128)) # match the input shape expected by the model

img_array = image.img_to_array(img) # to a numpy array

# Expand the dimensions to create a batch with a single sample (needed as the model expects a batch, even as single sample)
img_batch = np.expand_dims(img_array, axis=0) # (1, height, width, channels) needed for the model

prediction = model.predict(img_batch) # probability indicating the likelihood of the image containing a tumor
# Raw Output Prediction like:
# [[1.]]         suggests a high probability of tumor presence.
# [[0.00070843]] suggests a low probability of tumor presence.

threshold = 0.5 

print('The image is classified as likely to: ')
if prediction[0] > threshold:
    print('having tumor"')
else:
    print('not having tumor"')



import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# Load TFLite model and allocate tensors.
interpreter = tf.contrib.lite.Interpreter(model_path="converted_models/48-3-4-2-color-with-nan.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
# change the following line to feed into your own data.

img = Image.open('data/testing_data/32/32-113.jpg')
img = img.resize((48, 48))
np_img = np.array(img,dtype=np.float32).reshape(1,48,48,3)

interpreter.set_tensor(input_details[0]['index'], np_img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

loc = np.where(output_data==1)
labels = pd.read_csv('derp.txt')

print(str(loc))

print(output_data)

print(1)
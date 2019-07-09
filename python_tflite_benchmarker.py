import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from glob import glob
import numpy as np
from math import isnan


labels = pd.read_csv('derp.txt',names=['k','v'])

#def get_image_value(interpereter, img, img_size, labels):


def get_value_from_path(path):
    return os.path.dirname(path).split('/')[-1]


def run_benchmark(model_path = "converted_models/48-3-4-2-color-with-nan.tflite"):
    results_list = []
    # Load TFLite model and allocate tensors.
    interpreter = tf.contrib.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    files = glob('data/testing_data/*/*')
    for ix, file in enumerate(files):
        try:
            img = Image.open(file)
            img = img.resize((48, 48))
            np_img = np.array(img, dtype=np.float32).reshape(*input_shape)
            interpreter.set_tensor(input_details[0]['index'], np_img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            loc = np.where(output_data == 1)[1][0]
            res = labels.iloc[loc, 1]
            ref = get_value_from_path(file)
            if isnan(res):
                result = int(isnan(ref))
            else:
                result = int( int(ref)==int(res) )


            results_list.append(result)

            if ix%10==0:
                print(np.mean(results_list)*100)
        except:
            pass

if __name__=='__main__':
    run_benchmark()
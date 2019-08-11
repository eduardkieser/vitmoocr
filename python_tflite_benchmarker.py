
import tensorflow as tf
import os
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
import numpy as np
from math import isnan
from matplotlib import pyplot as plt
from random import shuffle


labels = pd.read_csv('derp.txt',names=['k','v'])

#def get_image_value(interpereter, img, img_size, labels):


def get_value_from_path(path):
    return os.path.dirname(path).split('/')[-1]


def run_benchmark(model_path="converted_models/36-3-4-2-color-with-nan.tflite"):
    results_list = []
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']

    files = glob('data/testing_data/*/*')
    shuffle(files)
    incorrect_classes = []
    all_classes = []

    for ix, file in enumerate(files):
        try:
            img = Image.open(file)
            img = img.resize((36, 36))
            np_img = np.array(img, dtype=np.float32).reshape(*input_shape)

            img_mean = np.mean(np_img)
            img_std = np.std(np_img)

            np_img = np_img/255
            interpreter.set_tensor(input_details[0]['index'], np_img)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            loc = np.where(output_data == 1)[1][0]
            res = labels.iloc[loc, 1]
            ref = get_value_from_path(file)


            all_classes.append({'ref':ref, 'res':res, 'im_mean':img_mean, 'im_std':img_std})

            if isnan(res):
                result = int(isnan(ref))
            else:
                result = int( int(ref)==int(res) )

            if ref=='nan':
                print('we have a nan')

            results_list.append(result)

            if not result:
                incorrect_classes.append({'ref':ref, 'res':res})

            if ix%1000==0:
                print(np.mean(results_list)*100)

            if ix >= 100000:
                break
        except:
            pass

    incorrect_classes_df = pd.DataFrame(incorrect_classes)
    incorrect_classes_df.to_csv('incorrect_classes.csv')
    all_classes_df = pd.DataFrame(all_classes)
    all_classes_df.to_csv('all_classes.csv')

def plot_results():
    all_classes_df = pd.read_csv('all_classes.csv', index_col=0)
    all_classes_df['delta'] = all_classes_df['res']-all_classes_df['ref']

    min_value_count = all_classes_df['ref'].value_counts().min()

    all_classes_balanced = all_classes_df.groupby('ref').head(min_value_count)

    incorrect_cases_ix = all_classes_balanced['ref']!=all_classes_balanced['res']
    incorrect_cases = all_classes_balanced[incorrect_cases_ix]
    # incorrect_cases['ref'].hist()

    all_classes_df['im_mean'].hist()

    j=''

if __name__=='__main__':
    run_benchmark()

    plot_results()
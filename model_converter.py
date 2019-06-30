import tensorflow as tf

# Save tf.keras model in HDF5 format.



input_file = "/home/eduard/workspace/VitmoOCR/trained_models/48-3-4-2-color-with-nan/ep12-va0.19.hdf5"

# Convert to TensorFlow Lite model.
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(input_file)
tflite_model = converter.convert()
open("converted_models/48-3-4-2-color-with-nan.tflite", "wb").write(tflite_model)
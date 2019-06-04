import numpy as np
import tensorflow as tf

def assemble_model(
    input_size = 28, 
    kernel_size=5, 
    n_conv_layers = 2, 
    dropout_rate = 0.1, 
    optimizer = tf.keras.optimizers.RMSprop(lr=0.0001))

    imput_shape = (imput_size, input_size, 1)
    kernel_shape = (kernel_size, kernel_size)

    model = tf.keras.models.Sequential()

    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv2D(
            n_features, 
            kernel_size=kernel_shape, 
            strides=(1, 1),
            activation='relu',
            input_shape=input_shape))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))    

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.categorical_accuracy],)

    return model

def assemble_data_generators(img_size = 28):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale=1./255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        directory=r"./tf_files/numbers_train/",
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale=1./255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=False)

    valid_generator = valid_datagen.flow_from_directory(
        directory=r"./tf_files/numbers_test/",
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    return train_generator, valid_generator

def train_model(model, train_generator, test_generator):
    


if __name__=='__main__':

num_classes = 201
input_sizes = [28, 32, 48]
kernel_sizes = [3, 5, 7]
dropout_rates = [0.05, 0.1, 0.2, 0.4]
n_features  = [16,32,64]
optimizers = [tf.keras.optimizers.SGD(lr=0.01), tf.keras.optimizers.RMSprop(lr=0.0001)]
n_conv_layers = [1,2,3]




STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
model.fit_generator(
    generator=train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=valid_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=10
)

# Save tf.keras model in HDF5 format.
keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
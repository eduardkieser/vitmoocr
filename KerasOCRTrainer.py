import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import os


def assemble_model(
    input_size = 28,
    n_features = 32,
    kernel_size=5,
    n_conv_layers = 2, 
    dropout_rate = 0.1, 
    optimizer = tf.keras.optimizers.RMSprop(lr=0.0001),
    num_classes = 201):

    input_shape = (input_size, input_size, 1)
    kernel_shape = (kernel_size, kernel_size)

    model = tf.keras.models.Sequential()

    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv2D(
            filters = n_features, 
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


def assemble_data_generators(img_size = 28, data_path_template='data/{}'):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale=1./255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        directory=data_path_template.format('training_data/'),
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            # rescale=1./255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=False)

    valid_generator = valid_datagen.flow_from_directory(
        directory=data_path_template.format('testing_data/'),
        target_size=(img_size, img_size),
        color_mode="grayscale",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    return train_generator, valid_generator


def train_model(model, train_generator, valid_generator, model_name, epocs):

    model_path = f'trained_models/{model_name}'
    if (not os.path.isdir(model_path)):
        os.makedirs(model_path)

    all_checkpoint_path = f'{model_path}''/ep{epoch:02d}-va{val_loss:.2f}.hdf5'
    save_all_callback = ModelCheckpoint(
        all_checkpoint_path, 
        monitor='val_loss',
        save_best_only=False,
    )
    best_checkpoint_path = f'{model_path}/best_so_far''{epoch:02d}-va{val_loss:.2f}.hdf5'
    save_best_callback = ModelCheckpoint(
        best_checkpoint_path, 
        monitor='val_loss',
        save_best_only=False,
    )

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    print('start training')
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=epocs,
        callbacks=[save_all_callback, save_best_callback]
    )


def run_model_optomization():

    num_classes = 201
    input_sizes = [28, 32, 48]
    kernel_sizes = [3, 5, 7]
    dropout_rates = [0.05, 0.1, 0.2, 0.4]
    n_features  = [16,32,64]
    optimizers = [
        tf.keras.optimizers.SGD(lr=0.01), 
        tf.keras.optimizers.SGD(lr=0.01, nesterov=True),
        tf.keras.optimizers.Adam(lr=0.001)]
    n_conv_layers = [1,2,3]

    input_size = input_sizes[0]
    kernel_size = kernel_sizes[0]
    dropout_rate = dropout_rates[0]
    optimizer = optimizers[1]

    # for for for...
    model = assemble_model(
        input_size = input_size, 
        n_features = n_features[0],
        kernel_size=kernel_size, 
        n_conv_layers = n_conv_layers[0], 
        dropout_rate = dropout_rate, 
        optimizer = optimizer,
        num_classes = 200)

    train_generator, valid_generator = \
        assemble_data_generators(img_size = input_size)

    model_name = f'{input_size}-{kernel_size}-{n_conv_layers}-{dropout_rate}-{optimizer}.hdf5'
    train_model(model, train_generator, valid_generator, model_name, epocs=5)
    
    
def convert_model_to_h5():
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)


if __name__=='__main__':


    run_model_optomization()

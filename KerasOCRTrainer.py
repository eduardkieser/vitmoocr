import sys
print(sys.executable)
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import os
from time import time
import pandas as pd


def assemble_model(
    input_size,
    n_features,
    kernel_size,
    stride_size,
    n_conv_layers, 
    dropout_rate, 
    optimizer,
    num_classes):

    input_shape = (input_size, input_size, 3)
    kernel_shape = (kernel_size, kernel_size)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=kernel_size,
                strides=(1, 1),
                activation='relu',
                input_shape=input_shape))

    for i in range(n_conv_layers):
        model.add(tf.keras.layers.Conv2D(
            filters = n_features, 
            kernel_size=kernel_shape, 
            strides=(stride_size, stride_size),
            activation='relu',
            input_shape=input_shape))
        # model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))    

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=[tf.keras.metrics.categorical_accuracy],)
    model.summary()

    return model


def assemble_data_generators(img_size, data_path_template='data/{}'):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=False)

    train_generator = train_datagen.flow_from_directory(
        directory=data_path_template.format('training_data/'),
        target_size=(img_size, img_size),
        # color_mode="grayscale",
        color_mode='rgb',
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            # zoom_range=0.2,
            horizontal_flip=False)

    valid_generator = valid_datagen.flow_from_directory(
        directory=data_path_template.format('testing_data/'),
        target_size=(img_size, img_size),
        # color_mode="grayscale",
        color_mode='rgb',
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42)

    return train_generator, valid_generator


def get_callbacks(model_name):
    model_path = f'trained_models/{model_name}'
    all_checkpoint_path = f'{model_path}''/ep{epoch:02d}-va{val_loss:.2f}.hdf5'
    print(all_checkpoint_path)
    save_all_callback = ModelCheckpoint(
        all_checkpoint_path, 
        monitor='val_loss',
        save_best_only=False,
    )
    best_checkpoint_path = f'{model_path}/best_so_far''{epoch:02d}-va{val_loss:.2f}.hdf5'
    print(best_checkpoint_path)
    save_best_callback = ModelCheckpoint(
        best_checkpoint_path, 
        monitor='val_loss',
        save_best_only=False,
    )

    # Create a TensorBoard instance with the path to the logs directory
    print(f'{model_name}')
    tensorboard = TensorBoard(log_dir=f'logs2/{model_name}')

    return tensorboard, save_all_callback


def create_labels_file(train_generator):
    labels = (train_generator.class_indices)
    # labels = dict((v,k) for k,v in labels.items())
    labels_lst = [{'v':v,'k':k} for v,k in labels.items()]

    labels_df = pd.DataFrame(labels_lst).set_index('k')['v']

    labels_df.to_csv('derp.txt')

    i=0


def train_model(model, train_generator, valid_generator, model_name, epocs):

    model_path = f'trained_models/{model_name}'
    if (not os.path.isdir(model_path)):
        os.makedirs(model_path)

    model.summary()

    i = 'go'

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
    print('start training')

    tensorboard, save_all_callback = get_callbacks(model_name)

    model.fit_generator(
        generator=train_generator,
        # steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        # validation_steps=STEP_SIZE_VALID,
        epochs=epocs,
        callbacks=[tensorboard, save_all_callback]
    )


def run_model_optomization():

    num_classes = 202
    input_size = 36
    kernel_size = 3
    dropout_rate = 0.2
    n_features  = 64
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    n_conv_layers = 3
    stride_size = 2

    # for for for...
    model = assemble_model(
        input_size = input_size, 
        n_features = n_features,
        kernel_size=kernel_size, 
        n_conv_layers = n_conv_layers, 
        stride_size = stride_size,
        dropout_rate = dropout_rate, 
        optimizer = optimizer,
        num_classes = num_classes)

    train_generator, valid_generator = \
        assemble_data_generators(img_size = input_size)

    create_labels_file(train_generator)

    model_name = f'{input_size}-{kernel_size}-{n_conv_layers+1}-{stride_size}-color-with-nan'

    train_model(model, train_generator, valid_generator, model_name, epocs=5)

    create_labels_file(train_generator)
    
    
def convert_model_to_h5():
    converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)



if __name__=='__main__':


    run_model_optomization()

    # launch tensorboard: in terminal
    # tensorboard --logdir=logs2/ --host localhost --port 8088
    # go to http://localhost:8088
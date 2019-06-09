import numpy as np
import tensorflow as tf

filepath = './model_checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

num_classes = 200

# usually 224
im_width = 48
im_height = 48

input_shape = (im_width, im_height, 3)

_depth = 64

conv_kernel_size = (3, 3)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(
                filters=32,
                kernel_size=conv_kernel_size,
                strides=(1, 1),
                activation='relu',
                input_shape=input_shape))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(tf.keras.layers.Conv2D(_depth, conv_kernel_size, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(_depth, conv_kernel_size, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(_depth, conv_kernel_size, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1000, activation='relu'))
# model.add(tf.keras.layers.Dense(1000, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))



model.compile(loss=tf.keras.losses.categorical_crossentropy,
              # optimizer=tf.keras.optimizers.SGD(lr=0.01),
                optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                # metrics=['accuracy']
                metrics=[tf.keras.metrics.categorical_accuracy],
                )


# model.compile(loss=tf.keras.losses.MSE,
#               optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
#               metrics=[tf.keras.metrics.categorical_accuracy],
#               sample_weight_mode='temporal')

#============== Setup data and use of flow from folder ================

# from example https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
    directory=r"./data/training_data/",
    target_size=(im_width, im_height),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"./data/testing_data/",
    target_size=(im_width, im_height),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42
)


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

model.summary()

go = True

if go:
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=valid_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=50,
        callbacks=[cp_callback]
    )



    # # Save tf.keras model in HDF5 format.
    # keras_file = "keras_model.h5"
    # tf.keras.models.save_model(model, keras_file)
    #
    # # Convert to TensorFlow Lite model.
    # converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
    # tflite_model = converter.convert()
    # open("converted_model.tflite", "wb").write(tflite_model)
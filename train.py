#
# Inspired from:
# https://github.com/dracoboros/Cats-Or-Dogs/blob/master/src/CatsOrDogs.ipynb
# https://www.dlology.com/blog/how-to-choose-last-layer-activation-and-loss-function/
# https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/5.2-using-convnets-with-small-datasets.ipynb
#
import datetime
import os

import tensorflow as tf


IMAGE_SIZE = 192
BATCH_SIZE = 32
EPOCHS = 10

DATASET_PATH = os.path.abspath("./dataset")
MODEL_PATH = os.path.abspath("./model")
LOG_TS = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

TRAIN_PATH = os.path.abspath(f"{DATASET_PATH}/train/")
VALIDATION_PATH = os.path.abspath(f"{DATASET_PATH}/validation/")

CLASSES = sorted(os.listdir(TRAIN_PATH))

################################################################################
# Data preparation
print("#" * 80)
print("Preparing datasets...")

train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255)
train_generator = train_datagenerator.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagenerator.flow_from_directory(
    VALIDATION_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

for data_batch, labels_batch in train_generator:
    print('Data batch shape:', data_batch.shape)
    print('Labels batch shape:', labels_batch.shape)
    break

################################################################################
# Model preparation
print("#" * 80)
print("Preparing model...")
tf.keras.backend.clear_session()
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(32, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same'))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Conv2D(64, (3, 3)))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512))
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(CLASSES), activation='softmax'))

################################################################################
# Model summary
print("#" * 80)
print("Model summary...")
model.summary()

################################################################################
# Model compile
print("#" * 80)
print("Compiling model...")
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

################################################################################
# Model train + tensorboard logs

print("#" * 80)
print("Training model...")
model.fit(train_generator,
          epochs=EPOCHS,
          steps_per_epoch=int(train_generator.samples / train_generator.batch_size),
          validation_data=validation_generator,
          validation_steps=int(validation_generator.samples / validation_generator.batch_size),
          callbacks=[tf.keras.callbacks.TensorBoard(log_dir="logs/fit/" + LOG_TS,
                                                    profile_batch=0, histogram_freq=0)])

################################################################################
# Labels save
LABEL_PATH = os.path.abspath(f'{MODEL_PATH}/model.labels')
print("#" * 80)
print("Saving labels [%s] to '%s'..." % ("|".join(CLASSES), LABEL_PATH))
with open(LABEL_PATH, 'w') as lf:
    lf.write('\n'.join(CLASSES))

################################################################################
# Model save
print("#" * 80)
print(f"Saving model to '{MODEL_PATH}'...")
model.save(MODEL_PATH)

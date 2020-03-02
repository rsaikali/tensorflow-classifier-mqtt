#
# Inspired from: https://github.com/dracoboros/Cats-Or-Dogs/blob/master/src/CatsOrDogs.ipynb
#
import os
from pathlib import Path

import tensorflow as tf

IMAGE_SIZE = 192
BATCH_SIZE = 128
EPOCHS = 5
STEPS_PER_EPOCHS = 500

TRAIN_PATH = os.path.abspath('./dataset/train/')
VALIDATION_PATH = os.path.abspath('./dataset/validation/')
MODEL_PATH = os.path.abspath('./model')
LABEL_PATH = os.path.abspath('./model/model.labels')
CLASSES = sorted(os.listdir(TRAIN_PATH))


################################################################################
# Data preparation
print("#" * 80)
print("Preparing datasets...")


def preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.image.rgb_to_grayscale(image)
    image /= 255.0
    return image


def load_dataset(path, shuffle=True):

    data_paths = []
    data_labels = []

    for item in Path(path).glob('**/*'):
        if item.is_file() and str(item).endswith('.jpg'):
            data_paths.append(str(item))
            data_labels.append(CLASSES.index(os.path.basename(os.path.dirname(str(item)))))

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(data_labels, tf.int64))

    path_ds = tf.data.Dataset.from_tensor_slices(data_paths)
    image_ds = path_ds.map(preprocess_image, num_parallel_calls=BATCH_SIZE)

    ds = tf.data.Dataset.zip((image_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=BATCH_SIZE).repeat()

    ds = ds.batch(BATCH_SIZE)

    return ds


train_ds = load_dataset(TRAIN_PATH, shuffle=True)
validation_ds = load_dataset(VALIDATION_PATH, shuffle=False)

################################################################################
# Model preparation
print("#" * 80)
print("Preparing model...")
tf.keras.backend.clear_session()
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
    tf.keras.layers.Dense(units=128, activation=tf.nn.leaky_relu),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')
])

################################################################################
# Model compile
print("#" * 80)
print("Compiling model...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.metrics.SparseCategoricalAccuracy()])

################################################################################
# Model train
print("#" * 80)
print("Training model...")
model.fit(train_ds, epochs=EPOCHS, steps_per_epoch=STEPS_PER_EPOCHS)

################################################################################
# Model train
print("#" * 80)
print("Evalutate model...")
model.evaluate(validation_ds)

################################################################################
# Labels save
print("#" * 80)
print("Saving labels [%s] to '%s'..." % ("|".join(CLASSES), LABEL_PATH))
with open(LABEL_PATH, 'w') as lf:
    lf.write('\n'.join(CLASSES))

################################################################################
# Model save
print("#" * 80)
print(f"Saving model to '{MODEL_PATH}'...")
model.save(MODEL_PATH)

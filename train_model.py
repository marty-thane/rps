from constants import *
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# normalize, introduce artificial variation
generator = ImageDataGenerator(
    rescale = 1 / 255,
    horizontal_flip = True,
    brightness_range = (0.8, 1.2),
    rotation_range = 10,
    zoom_range = (1, 1.2),
    validation_split = 0.2,
)

training = generator.flow_from_directory(
    DATA_DIR,
    target_size = (DATA_SIZE,DATA_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "categorical",
    color_mode = "grayscale",
    subset = "training"
)

validation = generator.flow_from_directory(
    DATA_DIR,
    target_size = (DATA_SIZE,DATA_SIZE),
    batch_size = BATCH_SIZE,
    class_mode = "categorical",
    color_mode = "grayscale",
    subset = "validation"
)

model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation = "relu", input_shape = (DATA_SIZE, DATA_SIZE, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation = "relu"))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation = "relu"))
# model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(32, activation = "relu"))
model.add(layers.Dense(4, activation = "softmax")) # empty, paper, rock, scissors

model.summary()

model.compile(optimizer = "adam",
              loss = "categorical_crossentropy",
              metrics = ["accuracy"]
)

model.fit(training,
          epochs = EPOCHS,
          validation_data = validation
)

model.save(f"{ASSET_DIR}/vision.keras")

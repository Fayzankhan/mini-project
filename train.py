import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# To remove warning logs in command prompt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use the reduced dataset (or full dataset)
DATASET_PATH = "dataset/"

# Set image size & training parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 10  # Reduce for faster training

# Data Augmentation (helps improve performance with fewer images)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Load Training & Validation Data
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# Load Pretrained MobileNetV2 Model (without top layers)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(128, 128, 3), include_top=False, weights="imagenet"
)
base_model.trainable = False  # Freeze pretrained layers

# Add Custom Layers (Fix Output Layer Issue)
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Fix flattening issue
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output_layer = Dense(1, activation='sigmoid')(x)  # Final output layer

# Define Model Properly
model = Model(inputs=base_model.input, outputs=output_layer)

# Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save Trained Model
os.makedirs("saved_model", exist_ok=True)  # Ensure folder exists
model.save("saved_model/pneumonia_detector.h5")

print("Model training complete! Saved as 'saved_model/pneumonia_detector.h5'")

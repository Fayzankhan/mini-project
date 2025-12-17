import os
import tensorflow as tf
import numpy as np
from model import create_cnn_lstm_model, create_sequence_generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# To remove warning logs in command prompt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Use the reduced dataset (or full dataset)
DATASET_PATH = "dataset/"

# Set image size & training parameters
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20  # Increased epochs for better learning
NUM_FRAMES = 4  # Number of frames in each sequence

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

# Create sequence generators
train_sequence_generator = create_sequence_generator(train_generator, NUM_FRAMES)
val_sequence_generator = create_sequence_generator(val_generator, NUM_FRAMES)

# Create and compile the CNN-LSTM model
model = create_cnn_lstm_model(input_shape=IMG_SIZE + (3,), num_frames=NUM_FRAMES)

# Print model summary
model.summary()

# Add callbacks for better training
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        'saved_model/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy'
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-6
    )
]

# Calculate steps per epoch
steps_per_epoch = train_generator.samples // BATCH_SIZE
validation_steps = val_generator.samples // BATCH_SIZE

# Train Model
history = model.fit(
    train_sequence_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_sequence_generator,
    validation_steps=validation_steps,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save Trained Model
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/pneumonia_detector_lstm.h5")

# Generate predictions for validation set
val_generator.reset()
y_true = []
y_pred = []

for i in range(validation_steps):
    x_batch, y_batch = next(val_sequence_generator)
    pred_batch = model.predict(x_batch)
    y_true.extend(y_batch)
    y_pred.extend((pred_batch > 0.5).astype(int))

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save confusion matrix plot
os.makedirs("static/plots", exist_ok=True)
plt.savefig('static/plots/confusion_matrix.png')
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

print("Model training complete! Saved as 'saved_model/pneumonia_detector_lstm.h5'")

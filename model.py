import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, LSTM, 
    Reshape, TimeDistributed, Conv2D, MaxPooling2D,
    BatchNormalization, Input
)

def create_cnn_lstm_model(input_shape=(128, 128, 3), num_frames=4):
    """
    Create a hybrid CNN-LSTM model for pneumonia detection.
    The model first extracts features using CNN layers, then processes temporal patterns using LSTM.
    
    Args:
        input_shape: Shape of each input frame (height, width, channels)
        num_frames: Number of consecutive frames to process
    """
    
    # Input layer for sequence of frames
    inputs = Input(shape=(num_frames,) + input_shape)
    
    # CNN Feature Extractor (wrapped in TimeDistributed)
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    # Flatten CNN output while keeping time dimension
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # LSTM layers for temporal processing
    x = LSTM(256, return_sequences=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)
    x = Dropout(0.3)(x)
    
    # Dense layers for classification
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create and compile model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_sequence_generator(base_generator, num_frames=4):
    """
    Create a generator that yields sequences of frames from base generator.
    This allows us to process multiple consecutive frames for temporal analysis.
    
    Args:
        base_generator: ImageDataGenerator for single frames
        num_frames: Number of consecutive frames to combine
    """
    while True:
        # Initialize empty lists for batch
        batch_images = []
        batch_labels = []
        
        # Get a batch of single frames
        images, labels = next(base_generator)
        batch_size = images.shape[0]
        
        # Create sequences by repeating frames (since we don't have true sequences)
        for i in range(batch_size):
            # Create sequence by duplicating the frame
            sequence = np.tile(images[i:i+1], (num_frames, 1, 1, 1))
            batch_images.append(sequence)
            batch_labels.append(labels[i])
        
        # Convert to numpy arrays
        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)
        
        yield batch_images, batch_labels 
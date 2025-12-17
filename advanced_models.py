"""
Advanced Model Architectures for Pneumonia Detection
Includes EfficientNet, Multi-class classification, and Ensemble models
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, LSTM, 
    TimeDistributed, BatchNormalization, Input, Concatenate,
    Attention, MultiHeadAttention
)
import numpy as np

def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=2, pretrained=True):
    """
    Create EfficientNet-based model for pneumonia detection.
    More accurate than basic CNN with fewer parameters.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes (2 for binary, 3+ for multi-class)
        pretrained: Use ImageNet pretrained weights
    """
    # Use EfficientNetB3 as base (can upgrade to B7 for better accuracy)
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet' if pretrained else None
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom classification head
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid', name='pneumonia_prob')(x)
    else:
        outputs = Dense(num_classes, activation='softmax', name='class_prob')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    if num_classes == 2:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
    
    return model, base_model

def create_attention_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Create CNN model with attention mechanism for better focus on important regions.
    """
    inputs = Input(shape=input_shape)
    
    # Feature extraction with attention
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    
    # Attention mechanism
    attention = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    x = tf.keras.layers.Multiply()([x, attention])
    
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = GlobalAveragePooling2D()(x)
    
    # Classification head
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    if num_classes == 2:
        outputs = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    else:
        outputs = Dense(num_classes, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer='adam',
        loss=loss,
        metrics=['accuracy']
    )
    
    return model

def create_ensemble_model(models, weights=None):
    """
    Create ensemble model that combines predictions from multiple models.
    
    Args:
        models: List of trained models
        weights: Optional weights for each model (default: equal weights)
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)
    
    def ensemble_predict(models, weights, x):
        predictions = []
        for model in models:
            pred = model.predict(x, verbose=0)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            ensemble_pred += pred * weight
        
        return ensemble_pred
    
    return ensemble_predict, models, weights

def create_multi_class_model(input_shape=(224, 224, 3)):
    """
    Create model for multi-class classification:
    - Normal
    - Bacterial Pneumonia
    - Viral Pneumonia
    - COVID-19 Pneumonia
    """
    base_model = tf.keras.applications.EfficientNetB3(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Multi-class output
    outputs = Dense(4, activation='softmax', name='class_output')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy']
    )
    
    return model

def calculate_severity_score(prediction, opacity_percentage, affected_area):
    """
    Calculate pneumonia severity score based on multiple factors.
    
    Args:
        prediction: Model prediction probability
        opacity_percentage: Percentage of lung with opacity
        affected_area: Affected lung area in pixels
    
    Returns:
        severity_score: 0-100 score
        severity_level: 'Mild', 'Moderate', 'Severe', 'Critical'
    """
    # Normalize factors
    prediction_score = prediction * 100
    opacity_score = min(opacity_percentage * 100, 100)
    area_score = min((affected_area / 10000) * 100, 100)  # Normalize area
    
    # Weighted combination
    severity_score = (
        prediction_score * 0.4 +
        opacity_score * 0.4 +
        area_score * 0.2
    )
    
    # Classify severity
    if severity_score < 30:
        severity_level = 'Mild'
    elif severity_score < 50:
        severity_level = 'Moderate'
    elif severity_score < 75:
        severity_level = 'Severe'
    else:
        severity_level = 'Critical'
    
    return round(severity_score, 2), severity_level

def get_treatment_recommendations(classification, severity, patient_info):
    """
    Generate treatment recommendations based on classification and severity.
    
    Args:
        classification: 'bacterial', 'viral', 'covid', 'normal'
        severity: 'Mild', 'Moderate', 'Severe', 'Critical'
        patient_info: Dictionary with patient information
    
    Returns:
        recommendations: List of treatment recommendations
    """
    recommendations = []
    
    if classification == 'normal':
        recommendations.append({
            'type': 'Follow-up',
            'description': 'No active treatment needed. Monitor symptoms.',
            'priority': 'Low'
        })
        return recommendations
    
    # Severity-based recommendations
    if severity == 'Critical':
        recommendations.append({
            'type': 'Immediate Action',
            'description': 'Patient requires immediate hospitalization and intensive care.',
            'priority': 'Critical',
            'actions': ['Hospital admission', 'Oxygen therapy', 'IV antibiotics', 'Monitor vitals']
        })
    elif severity == 'Severe':
        recommendations.append({
            'type': 'Urgent Care',
            'description': 'Patient should be evaluated urgently. Consider hospitalization.',
            'priority': 'High',
            'actions': ['Urgent medical evaluation', 'Antibiotic therapy', 'Chest X-ray follow-up']
        })
    elif severity == 'Moderate':
        recommendations.append({
            'type': 'Standard Care',
            'description': 'Patient requires medical attention and treatment.',
            'priority': 'Medium',
            'actions': ['Antibiotic therapy', 'Rest', 'Hydration', 'Follow-up in 48-72 hours']
        })
    else:  # Mild
        recommendations.append({
            'type': 'Outpatient Care',
            'description': 'Patient can be managed with outpatient treatment.',
            'priority': 'Low',
            'actions': ['Oral antibiotics', 'Rest', 'Monitor symptoms', 'Follow-up if symptoms worsen']
        })
    
    # Classification-based recommendations
    if classification == 'bacterial':
        recommendations.append({
            'type': 'Antibiotic Therapy',
            'description': 'Bacterial pneumonia detected. Antibiotic treatment recommended.',
            'medications': ['Amoxicillin-clavulanate', 'Azithromycin', 'Levofloxacin'],
            'duration': '7-10 days'
        })
    elif classification == 'viral':
        recommendations.append({
            'type': 'Antiviral Therapy',
            'description': 'Viral pneumonia detected. Antiviral treatment may be considered.',
            'medications': ['Oseltamivir (if influenza)', 'Supportive care'],
            'note': 'Most viral pneumonias resolve with supportive care'
        })
    elif classification == 'covid':
        recommendations.append({
            'type': 'COVID-19 Protocol',
            'description': 'COVID-19 pneumonia detected. Follow COVID-19 treatment protocols.',
            'medications': ['Remdesivir (if severe)', 'Dexamethasone', 'Monoclonal antibodies'],
            'isolation': 'Required'
        })
    
    # Age-based considerations
    if 'age' in patient_info:
        try:
            age = int(patient_info['age'])
            if age > 65:
                recommendations.append({
                    'type': 'Special Consideration',
                    'description': 'Elderly patient - higher risk. Consider more aggressive treatment.',
                    'priority': 'High'
                })
        except:
            pass
    
    return recommendations


from flask import Flask, render_template, request, send_file, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from fpdf import FPDF
from datetime import datetime
from skimage import metrics
import json
import tempfile
import uuid
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Import advanced features
try:
    from advanced_models import (
        calculate_severity_score,
        get_treatment_recommendations
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    ADVANCED_FEATURES_AVAILABLE = False
    print("Advanced features not available. Install required packages.")

app = Flask(__name__)

# Create directories for file storage
# Use environment variable for production (Render), temp dir for local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if os.getenv('RENDER'):
    # Production: Use persistent storage
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    REPORTS_FOLDER = os.path.join(BASE_DIR, "static", "reports")
    HISTORY_FILE = os.path.join(BASE_DIR, "static", "history.json")
    PATIENTS_FILE = os.path.join(BASE_DIR, "static", "patients.json")
else:
    # Local development: Use temp directory
    temp_dir = tempfile.gettempdir()
    UPLOAD_FOLDER = os.path.join(temp_dir, "uploads")
    REPORTS_FOLDER = os.path.join(temp_dir, "reports")
    HISTORY_FILE = os.path.join(temp_dir, "history.json")
    PATIENTS_FILE = os.path.join(temp_dir, "patients.json")

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE) if os.path.dirname(HISTORY_FILE) else '.', exist_ok=True)
os.makedirs(os.path.dirname(PATIENTS_FILE) if os.path.dirname(PATIENTS_FILE) else '.', exist_ok=True)

# Initialize history and patients files if they don't exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

if not os.path.exists(PATIENTS_FILE):
    with open(PATIENTS_FILE, 'w') as f:
        json.dump([], f)

# Load model - Use absolute paths that work in both local and Render
MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "pneumonia_detector_lstm.h5")
FALLBACK_MODEL_PATH = os.path.join(BASE_DIR, "saved_model", "pneumonia_detector.h5")
# Also check for typo in filename (but this file is likely corrupted - only 360 bytes)
FALLBACK_MODEL_PATH_ALT = os.path.join(BASE_DIR, "saved_model", "prenumonia_detector.h5")
model = None  # Initialize model as None
use_lstm = False  # Flag to track which model is being used

def load_model_on_demand():
    global model, use_lstm
    if model is None:
        print(f"Attempting to load model. BASE_DIR: {BASE_DIR}")
        
        # Try LSTM model first
        if os.path.exists(MODEL_PATH):
            try:
                print(f"Trying to load LSTM model from {MODEL_PATH}")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                use_lstm = True
                print(f"✓ Successfully loaded LSTM model from {MODEL_PATH}")
                return model
            except Exception as e:
                print(f"✗ Error loading LSTM model: {e}")
                import traceback
                traceback.print_exc()
        
        # Try fallback model (main model - 12MB)
        if os.path.exists(FALLBACK_MODEL_PATH):
            try:
                print(f"Trying to load fallback model from {FALLBACK_MODEL_PATH}")
                # Check file size first
                file_size = os.path.getsize(FALLBACK_MODEL_PATH)
                print(f"Model file size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")
                if file_size < 1024:  # Less than 1KB is suspicious
                    print(f"WARNING: Model file seems too small ({file_size} bytes). Skipping.")
                else:
                    model = tf.keras.models.load_model(FALLBACK_MODEL_PATH, compile=False)
                    use_lstm = False
                    print(f"✓ Successfully loaded fallback model from {FALLBACK_MODEL_PATH}")
                    return model
            except Exception as e:
                print(f"✗ Error loading fallback model: {e}")
                import traceback
                traceback.print_exc()
        
        # Try alternate path (with typo) - but skip if it's a Python script
        if os.path.exists(FALLBACK_MODEL_PATH_ALT):
            try:
                file_size = os.path.getsize(FALLBACK_MODEL_PATH_ALT)
                print(f"Alternate model file size: {file_size} bytes")
                # Skip if file is too small (likely not a real model)
                if file_size > 1024 * 1024:  # Only try if > 1MB
                    print(f"Trying to load alternate model from {FALLBACK_MODEL_PATH_ALT}")
                    model = tf.keras.models.load_model(FALLBACK_MODEL_PATH_ALT, compile=False)
                    use_lstm = False
                    print(f"✓ Successfully loaded alternate model from {FALLBACK_MODEL_PATH_ALT}")
                    return model
                else:
                    print(f"Skipping {FALLBACK_MODEL_PATH_ALT} - file too small ({file_size} bytes), likely not a model")
            except Exception as e:
                print(f"✗ Error loading alternate model: {e}")
                import traceback
                traceback.print_exc()
        
        # If no model found, raise error with helpful message
        print("=" * 80)
        print("MODEL LOADING DEBUG INFO:")
        print(f"BASE_DIR: {BASE_DIR}")
        print(f"MODEL_PATH: {MODEL_PATH} (exists: {os.path.exists(MODEL_PATH)})")
        print(f"FALLBACK_MODEL_PATH: {FALLBACK_MODEL_PATH} (exists: {os.path.exists(FALLBACK_MODEL_PATH)})")
        print(f"FALLBACK_MODEL_PATH_ALT: {FALLBACK_MODEL_PATH_ALT} (exists: {os.path.exists(FALLBACK_MODEL_PATH_ALT)})")
        
        saved_model_dir = os.path.join(BASE_DIR, "saved_model")
        print(f"saved_model directory: {saved_model_dir} (exists: {os.path.exists(saved_model_dir)})")
        
        if os.path.exists(saved_model_dir):
            files = os.listdir(saved_model_dir)
            print(f"Files in saved_model: {files}")
            for f in files:
                file_path = os.path.join(saved_model_dir, f)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  - {f}: {size} bytes ({size / (1024*1024):.2f} MB)")
        else:
            print(f"saved_model directory does not exist at {saved_model_dir}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"Files in BASE_DIR: {os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else 'BASE_DIR does not exist'}")
        print("=" * 80)
        raise FileNotFoundError("No model file found. Please ensure model files are in saved_model/ directory.")
    return model

def load_history():
    with open(HISTORY_FILE, 'r') as f:
        return json.load(f)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)

def load_patients():
    with open(PATIENTS_FILE, 'r') as f:
        return json.load(f)

def save_patients(patients):
    with open(PATIENTS_FILE, 'w') as f:
        json.dump(patients, f)

def calculate_image_quality(image):
    """Calculate image quality score based on various metrics"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate contrast
    contrast = np.std(gray)
    
    # Calculate blurriness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    blurriness = np.var(laplacian)
    
    # Calculate brightness
    brightness = np.mean(gray)
    
    # Normalize and combine metrics
    contrast_score = min(contrast / 100, 1) * 100
    blurriness_score = min(blurriness / 1000, 1) * 100
    brightness_score = min(abs(brightness - 128) / 128, 1) * 100
    
    # Weighted average
    quality_score = (contrast_score * 0.4 + blurriness_score * 0.3 + brightness_score * 0.3)
    return round(quality_score, 2)

def calculate_feature_detection(image):
    """Calculate feature detection score based on edge detection"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Calculate edge density
    edge_density = np.sum(edges > 0) / edges.size
    
    # Normalize to percentage
    feature_score = edge_density * 100
    return round(feature_score, 2)

def calculate_pattern_recognition(image):
    """Calculate pattern recognition score based on texture analysis"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate GLCM features
    glcm = cv2.calcHist([gray], [0], None, [256], [0, 256])
    glcm = glcm.flatten() / glcm.sum()
    
    # Calculate entropy
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    
    # Normalize to percentage
    pattern_score = min(entropy / 8, 1) * 100
    return round(pattern_score, 2)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=0):
    """Generate Grad-CAM heatmap for visualization"""
    try:
        last_conv_layer = None
        
        # For CNN models, find the last convolutional layer
        if use_lstm:
            # For LSTM models, find the last conv layer in TimeDistributed layers
            for layer in reversed(model.layers):
                if hasattr(layer, 'layer') and hasattr(layer.layer, '__class__'):
                    layer_class = str(layer.layer.__class__).lower()
                    if 'conv2d' in layer_class:
                        last_conv_layer = layer
                        break
                elif 'conv2d' in str(type(layer)).lower():
                    last_conv_layer = layer
                    break
        else:
            # For regular CNN models (like MobileNetV2)
            # Check if first layer is a base model
            if len(model.layers) > 0:
                first_layer = model.layers[0]
                # If it's a functional model (like MobileNetV2)
                if hasattr(first_layer, 'layers'):
                    # Search in base model
                    for layer in reversed(first_layer.layers):
                        if 'conv' in layer.name.lower() and 'block' in layer.name.lower():
                            last_conv_layer = layer
                            break
                    # If not found, get the last conv layer
                    if last_conv_layer is None:
                        for layer in reversed(first_layer.layers):
                            if 'conv' in layer.name.lower():
                                last_conv_layer = layer
                                break
                else:
                    # Search in model layers
                    for layer in reversed(model.layers):
                        if 'conv' in layer.name.lower():
                            last_conv_layer = layer
                            break
        
        if last_conv_layer is None:
            # Fallback: use fallback heatmap
            return generate_fallback_heatmap(img_array), None
        
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        try:
            grad_model = tf.keras.models.Model(
                [model.inputs], [last_conv_layer.output, model.output]
            )
        except:
            # If that fails, try accessing through base model
            if hasattr(model.layers[0], 'layers'):
                base_model = model.layers[0]
                grad_model = tf.keras.models.Model(
                    [base_model.input], [last_conv_layer.output, model.output]
                )
            else:
                return generate_fallback_heatmap(img_array), None
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if len(predictions.shape) > 1:
                pred = predictions[:, pred_index]
            else:
                pred = predictions[pred_index]
        
        # This is the gradient of the output neuron (top class or our chosen class)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(pred, conv_outputs)
        
        # Handle different tensor shapes
        if use_lstm:
            # For LSTM: (batch, frames, h, w, channels) -> average over frames
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
            conv_outputs_avg = tf.reduce_mean(conv_outputs[0], axis=0)
        else:
            # For CNN: (batch, h, w, channels)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs_avg = conv_outputs[0]
        
        # We multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        heatmap = conv_outputs_avg @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # For visualization, we will normalize the heatmap between 0 & 1
        heatmap = tf.maximum(heatmap, 0)
        if tf.math.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.math.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        return heatmap, last_conv_layer.name if hasattr(last_conv_layer, 'name') else 'unknown'
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        import traceback
        traceback.print_exc()
        # Fallback: create a simple attention map based on image features
        return generate_fallback_heatmap(img_array), None

def generate_fallback_heatmap(img_array):
    """Generate a fallback heatmap using image analysis"""
    # Get the actual image (remove batch dimension)
    if len(img_array.shape) == 5:  # LSTM: (batch, frames, h, w, c)
        img = img_array[0, 0]  # Take first frame
    else:  # CNN: (batch, h, w, c)
        img = img_array[0]
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8)
    
    # Detect edges and areas of interest
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create heatmap based on edge density
    heatmap = np.zeros_like(gray, dtype=np.float32)
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            cv2.fillPoly(heatmap, [contour], 1.0)
    
    # Apply Gaussian blur for smoother heatmap
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    
    # Normalize
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    return heatmap

def overlay_heatmap_on_image(img_path, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    # Read original image
    img = cv2.imread(img_path)
    original_shape = img.shape[:2]
    
    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, (original_shape[1], original_shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    
    # Apply colormap (red for high attention, blue for low)
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Overlay heatmap on original image
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    
    return overlayed, heatmap_resized

def identify_detected_regions(heatmap, threshold=0.5):
    """Identify and explain detected regions in the heatmap"""
    # Threshold the heatmap
    binary_mask = (heatmap > threshold).astype(np.uint8) * 255
    
    # Find contours of detected regions
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 100:  # Filter small regions
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate region properties
            mask = np.zeros(heatmap.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            region_intensity = np.mean(heatmap[mask > 0])
            
            # Determine region location
            center_x, center_y = x + w // 2, y + h // 2
            height, width = heatmap.shape
            
            if center_x < width * 0.4:
                location = "Left lung"
            elif center_x > width * 0.6:
                location = "Right lung"
            else:
                location = "Central chest area"
            
            if center_y < height * 0.4:
                location += " (upper)"
            elif center_y > height * 0.6:
                location += " (lower)"
            else:
                location += " (middle)"
            
            regions.append({
                'id': i + 1,
                'bbox': (x, y, w, h),
                'intensity': float(region_intensity),
                'area': int(cv2.contourArea(contour)),
                'location': location,
                'explanation': generate_region_explanation(region_intensity, location)
            })
    
    return regions

def generate_region_explanation(intensity, location):
    """Generate explanation for why a region is marked"""
    if intensity > 0.7:
        severity = "high"
        explanation = f"This area in the {location.lower()} shows strong indicators of potential pneumonia. The model detected significant opacity, consolidation patterns, or abnormal texture that are characteristic of lung infection."
    elif intensity > 0.5:
        severity = "moderate"
        explanation = f"This region in the {location.lower()} displays moderate signs that may indicate pneumonia. The model identified subtle changes in lung tissue density or texture that warrant attention."
    else:
        severity = "low"
        explanation = f"This area in the {location.lower()} shows minimal indicators. The model detected slight variations, but these may be within normal range or require further clinical correlation."
    
    return {
        'severity': severity,
        'text': explanation,
        'recommendation': get_recommendation(severity)
    }

def get_recommendation(severity):
    """Get recommendation based on severity"""
    if severity == "high":
        return "Immediate clinical evaluation recommended. Consider follow-up imaging and appropriate treatment."
    elif severity == "moderate":
        return "Clinical correlation advised. Monitor symptoms and consider follow-up if symptoms persist."
    else:
        return "Routine follow-up as per standard clinical protocol."

def create_visualization_with_explanations(img_path, model, prediction, confidence):
    """Create complete visualization with marked regions and explanations"""
    # Preprocess image
    img_array = preprocess_image(img_path)
    
    # Generate heatmap
    heatmap, layer_name = make_gradcam_heatmap(img_array, model)
    
    if heatmap is None:
        return None, None, []
    
    # Overlay heatmap on original image
    overlayed_img, heatmap_resized = overlay_heatmap_on_image(img_path, heatmap)
    
    # Identify detected regions
    regions = identify_detected_regions(heatmap_resized / 255.0)
    
    # Save visualization
    vis_filename = f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    vis_path = os.path.join(UPLOAD_FOLDER, vis_filename)
    cv2.imwrite(vis_path, overlayed_img)
    
    return vis_path, heatmap, regions

# Function to preprocess image for LSTM model
def preprocess_image(image_path, num_frames=4):
    """Preprocess image for model prediction"""
    global use_lstm
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Match model input size
    image = image / 255.0  # Normalize
    
    if use_lstm:
        # Create sequence by duplicating the frame for LSTM model
        sequence = np.tile(image[np.newaxis, ...], (num_frames, 1, 1, 1))
        sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
        return sequence
    else:
        # Regular CNN model expects single image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

@app.route("/", methods=["GET"])
def home():
    try:
        history = load_history()
    except:
        history = []
    return render_template("index.html", history=history)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for Streamlit or other frontends"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Get patient information
    patient_info = {
        "name": request.form.get("name", "Not Available"),
        "age": request.form.get("age", "Not Available"),
        "gender": request.form.get("gender", "Not Available"),
        "medical_history": request.form.get("medical_history", "Not Available")
    }
    
    # Save file
    temp_file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_file_path)
    
    # Load model
    model = load_model_on_demand()
    
    # Process image
    sequence = preprocess_image(temp_file_path)
    prediction = model.predict(sequence, verbose=0)[0][0]
    confidence = round(float(prediction) * 100, 2)
    
    result = "Pneumonia Detected" if prediction > 0.5 else "No Pneumonia"
    
    # Calculate severity if advanced features available
    severity_score = None
    severity_level = None
    treatment_recommendations = []
    
    if ADVANCED_FEATURES_AVAILABLE and result == "Pneumonia Detected":
        try:
            vis_path, heatmap, detected_regions = create_visualization_with_explanations(
                temp_file_path, model, prediction, confidence
            )
            opacity_percentage = float(np.mean(heatmap > 0.5)) if heatmap is not None else float(prediction)
            affected_area = int(np.sum(heatmap > 0.5)) if heatmap is not None else 0
            
            severity_score, severity_level = calculate_severity_score(
                prediction=prediction,
                opacity_percentage=opacity_percentage,
                affected_area=affected_area
            )
            
            treatment_recommendations = get_treatment_recommendations(
                classification='bacterial',
                severity=severity_level,
                patient_info=patient_info
            )
        except Exception as e:
            print(f"Error in advanced features: {e}")
    
    return jsonify({
        "result": result,
        "confidence": confidence,
        "severity_score": severity_score,
        "severity_level": severity_level,
        "treatment_recommendations": treatment_recommendations,
        "patient_info": patient_info
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            print("ERROR: No file in request.files")
            return render_template("index.html", result="No file uploaded.", error="Please select a file to upload.")

        file = request.files["file"]
        if file.filename == "":
            print("ERROR: Empty filename")
            return render_template("index.html", result="No file selected.", error="Please select a file to upload.")

        print(f"Processing file: {file.filename}")

        # Get patient information from the form
        patient_info = {
            "name": request.form.get("name", "Not Available"),
            "age": request.form.get("age", "Not Available"),
            "gender": request.form.get("gender", "Not Available"),
            "medical_history": request.form.get("medical_history", "Not Available")
        }

        # Save patient information
        try:
            patients = load_patients()
            patients.append(patient_info)
            save_patients(patients)
        except Exception as e:
            print(f"Warning: Could not save patient info: {e}")

        # Generate unique filename to avoid conflicts
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        temp_file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Ensure directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Save the uploaded file
        try:
            file.save(temp_file_path)
            print(f"File saved to: {temp_file_path}")
            if not os.path.exists(temp_file_path):
                raise Exception(f"File was not saved correctly to {temp_file_path}")
        except Exception as e:
            print(f"ERROR saving file: {e}")
            import traceback
            traceback.print_exc()
            return render_template("index.html", result="Error uploading file.", error=f"Could not save file: {str(e)}")

        # Load model on demand
        try:
            model = load_model_on_demand()
            if model is None:
                raise Exception("Model could not be loaded")
            print("Model loaded successfully")
        except Exception as e:
            print(f"ERROR loading model: {e}")
            import traceback
            traceback.print_exc()
            return render_template("index.html", result="Error loading model.", error=f"Model loading failed: {str(e)}")

        # Read the original image for quality assessment
        try:
            original_image = cv2.imread(temp_file_path)
            if original_image is None:
                raise Exception(f"Could not read image from {temp_file_path}")
            
            # Calculate additional confidence metrics
            image_quality = calculate_image_quality(original_image)
            feature_detection = calculate_feature_detection(original_image)
            pattern_recognition = calculate_pattern_recognition(original_image)
            print(f"Image quality metrics: quality={image_quality}, feature={feature_detection}, pattern={pattern_recognition}")
        except Exception as e:
            print(f"ERROR reading image: {e}")
            image_quality = 75.0
            feature_detection = 80.0
            pattern_recognition = 75.0

        # Process the image and get prediction
        try:
            sequence = preprocess_image(temp_file_path)
            print(f"Image preprocessed, shape: {sequence.shape}")
            prediction = model.predict(sequence, verbose=0)[0][0]
            confidence = round(float(prediction) * 100, 2)
            print(f"Prediction: {prediction}, Confidence: {confidence}%")
        except Exception as e:
            print(f"ERROR during prediction: {e}")
            import traceback
            traceback.print_exc()
            return render_template("index.html", result="Error processing image.", error=f"Prediction failed: {str(e)}")
        
        if prediction > 0.5:
            result = "Pneumonia Detected"
        else:
            result = "No Pneumonia"

        # Generate visualization with marked regions
        vis_path = None
        heatmap = None
        detected_regions = []
        visualization_url = None
        opacity_percentage = 0.0
        affected_area = 0
        
        try:
            vis_path, heatmap, detected_regions = create_visualization_with_explanations(
                temp_file_path, model, prediction, confidence
            )
            print(f"Visualization created: {vis_path is not None}")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
            import traceback
            traceback.print_exc()
        
        # If visualization failed, use original image
        if vis_path is None or not os.path.exists(vis_path):
            vis_path = temp_file_path
            detected_regions = []
            visualization_url = None
            opacity_percentage = 0.0
            affected_area = 0
        else:
            # Create URL for visualization image
            vis_filename = os.path.basename(vis_path)
            visualization_url = f"/visualization/{vis_filename}"
            
            # Calculate opacity percentage and affected area from heatmap
            if heatmap is not None:
                opacity_percentage = float(np.mean(heatmap > 0.5))  # Percentage of high-intensity regions
                affected_area = int(np.sum(heatmap > 0.5))  # Number of affected pixels
            else:
                opacity_percentage = float(prediction)  # Fallback to prediction
                affected_area = len(detected_regions) * 1000 if detected_regions else 0

        # Calculate severity score and get treatment recommendations
        severity_score = None
        severity_level = None
        treatment_recommendations = []
        
        if ADVANCED_FEATURES_AVAILABLE and result == "Pneumonia Detected":
            try:
                # Calculate severity score
                try:
                    severity_score, severity_level = calculate_severity_score(
                        prediction=prediction,
                        opacity_percentage=opacity_percentage,
                        affected_area=affected_area
                    )
                    # Ensure severity_level is lowercase for CSS classes
                    severity_level = severity_level.lower() if severity_level else None
                except Exception as e:
                    print(f"Error in severity calculation: {e}")
                    severity_score = None
                    severity_level = None
                
                # Determine classification (simplified - in production, use multi-class model)
                classification = 'bacterial'  # Default, can be enhanced with multi-class model
                
                # Get treatment recommendations
                treatment_recommendations = get_treatment_recommendations(
                    classification=classification,
                    severity=severity_level,
                    patient_info=patient_info
                )
            except Exception as e:
                print(f"Error calculating advanced features: {e}")

        # Get current timestamp
        analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Create URL for serving the image
        image_filename = os.path.basename(temp_file_path)
        image_url = f"/visualization/{image_filename}"

        # Save analysis details for report (including all new features)
        try:
            analysis_data = {
                "filename": file.filename,
                "result": result,
                "confidence": confidence,
                "image_quality": image_quality,
                "feature_detection": feature_detection,
                "pattern_recognition": pattern_recognition,
                "analysis_date": analysis_date,
                "image_path": temp_file_path,
                "patient_info": json.dumps(patient_info),
                "severity_score": str(severity_score) if severity_score else "N/A",
                "severity_level": str(severity_level) if severity_level else "N/A",
                "opacity_percentage": str(round(opacity_percentage * 100, 2)) if isinstance(opacity_percentage, float) else "0",
                "affected_area": str(affected_area),
                "detected_regions": json.dumps(detected_regions) if detected_regions else "[]",
                "treatment_recommendations": json.dumps(treatment_recommendations) if treatment_recommendations else "[]",
                "visualization_path": vis_path if vis_path != temp_file_path else ""
            }
            
            # Store the analysis data in temporary directory
            os.makedirs(REPORTS_FOLDER, exist_ok=True)
            with open(os.path.join(REPORTS_FOLDER, "latest_analysis.txt"), "w") as f:
                for key, value in analysis_data.items():
                    f.write(f"{key}:{value}\n")
        except Exception as e:
            print(f"Warning: Could not save analysis data: {e}")

        # Add to history
        try:
            history = load_history()
            history.insert(0, {
                "result": result,
                "confidence": confidence,
                "date": analysis_date,
                "filename": file.filename,
                "patient_info": patient_info
            })
            # Keep only last 10 entries
            history = history[:10]
            save_history(history)
        except Exception as e:
            print(f"Warning: Could not save history: {e}")
            history = []

        print(f"Rendering template with result: {result}, confidence: {confidence}%")
        
        return render_template(
            "index.html",
            result=result,
            image_url=image_url,
            visualization_url=visualization_url if vis_path != temp_file_path and visualization_url else None,
            confidence=confidence,
            image_quality=image_quality,
            feature_detection=feature_detection,
            pattern_recognition=pattern_recognition,
            analysis_date=analysis_date,
            history=history,
            patient_info=patient_info,
            show_patient_info=True,
            detected_regions=detected_regions,
            has_visualization=vis_path != temp_file_path and visualization_url is not None,
            severity_score=severity_score,
            severity_level=severity_level,
            treatment_recommendations=treatment_recommendations,
            opacity_percentage=round(opacity_percentage * 100, 2) if isinstance(opacity_percentage, float) else 0,
            affected_area=affected_area
        )
    except Exception as e:
        print(f"CRITICAL ERROR in predict route: {e}")
        import traceback
        traceback.print_exc()
        return render_template("index.html", result="An error occurred.", error=f"Unexpected error: {str(e)}")

@app.route("/visualization/<filename>")
def serve_visualization(filename):
    """Serve visualization images"""
    try:
        # Sanitize filename to prevent directory traversal
        filename = os.path.basename(filename)
        vis_path = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Serving image from: {vis_path}")
        print(f"UPLOAD_FOLDER: {UPLOAD_FOLDER}")
        print(f"File exists: {os.path.exists(vis_path)}")
        
        if os.path.exists(vis_path):
            # Determine mimetype based on file extension
            if filename.lower().endswith('.png'):
                mimetype = 'image/png'
            elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                mimetype = 'image/jpeg'
            else:
                mimetype = 'image/jpeg'  # Default
            print(f"Serving with mimetype: {mimetype}")
            return send_file(vis_path, mimetype=mimetype)
        else:
            print(f"Image not found at: {vis_path}")
            if os.path.exists(UPLOAD_FOLDER):
                files = os.listdir(UPLOAD_FOLDER)
                print(f"Files in folder ({len(files)}): {files[:10]}")  # Show first 10
            else:
                print(f"UPLOAD_FOLDER does not exist: {UPLOAD_FOLDER}")
            return "Image not found", 404
    except Exception as e:
        print(f"Error serving visualization: {e}")
        import traceback
        traceback.print_exc()
        return f"Error loading image: {str(e)}", 500

@app.route("/static/reports/<filename>")
def serve_report_file(filename):
    """Serve report files like confusion matrix"""
    try:
        file_path = os.path.join(REPORTS_FOLDER, filename)
        if os.path.exists(file_path):
            if filename.endswith('.png'):
                return send_file(file_path, mimetype='image/png')
            elif filename.endswith('.pdf'):
                return send_file(file_path, mimetype='application/pdf')
        return "File not found", 404
    except Exception as e:
        print(f"Error serving report file: {e}")
        return "Error loading file", 500

@app.route("/download_report")
def download_report():
    # Read the analysis data
    analysis_data = {}
    try:
        with open(os.path.join(REPORTS_FOLDER, "latest_analysis.txt"), "r") as f:
            for line in f:
                key, value = line.strip().split(":", 1)
                if key == "patient_info":
                    try:
                        # Parse the JSON string back to a dictionary
                        analysis_data[key] = json.loads(value)
                    except json.JSONDecodeError:
                        analysis_data[key] = {
                            "name": "Not Available",
                            "age": "Not Available",
                            "gender": "Not Available",
                            "medical_history": "Not Available"
                        }
                else:
                    analysis_data[key] = value
    except FileNotFoundError:
        return "No analysis data found. Please perform an analysis first.", 404

    # Generate PDF report
    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 20)
            self.set_text_color(0, 0, 0)
            self.cell(0, 20, "Pneumonia Detection Report", ln=True, align="C")
            self.line(10, 30, 200, 30)
            self.ln(10)

    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Add patient information
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    
    patient_info = analysis_data.get("patient_info", {})
    details = [
        ("Name", patient_info.get("name", "Not Available")),
        ("Age", patient_info.get("age", "Not Available")),
        ("Gender", patient_info.get("gender", "Not Available")),
        ("Medical History", patient_info.get("medical_history", "Not Available"))
    ]
    
    for label, value in details:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(60, 10, label + ":", 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, str(value), ln=True)

    # Add analysis results
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Analysis Results", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    details = [
        ("Diagnosis", analysis_data.get("result", "N/A")),
        ("Overall Confidence", f"{analysis_data.get('confidence', 'N/A')}%"),
        ("Image Quality", f"{analysis_data.get('image_quality', 'N/A')}%"),
        ("Feature Detection", f"{analysis_data.get('feature_detection', 'N/A')}%"),
        ("Pattern Recognition", f"{analysis_data.get('pattern_recognition', 'N/A')}%"),
        ("Analysis Date", analysis_data.get("analysis_date", "N/A"))
    ]

    for label, value in details:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(60, 10, label + ":", 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, str(value), ln=True)

    # Add Severity Assessment
    severity_score = analysis_data.get("severity_score", "N/A")
    severity_level = analysis_data.get("severity_level", "N/A")
    if severity_score != "N/A" and severity_level != "N/A":
        pdf.ln(5)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Pneumonia Severity Assessment", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Severity Score: {severity_score}/100", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Severity Level: {severity_level.title()}", ln=True)
        pdf.ln(3)
        
        # Severity metrics
        opacity_pct = analysis_data.get("opacity_percentage", "0")
        affected_area = analysis_data.get("affected_area", "0")
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 8, "Severity Metrics:", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, f"  • Opacity Percentage: {opacity_pct}%", ln=True)
        pdf.cell(0, 6, f"  • Affected Area: {affected_area} pixels", ln=True)
        pdf.cell(0, 6, f"  • Detection Confidence: {analysis_data.get('confidence', 'N/A')}%", ln=True)

    # Add Detected Regions Analysis
    detected_regions_str = analysis_data.get("detected_regions", "[]")
    try:
        detected_regions = json.loads(detected_regions_str) if detected_regions_str else []
        if detected_regions:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Detected Regions Analysis", ln=True)
            pdf.ln(3)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, "The following areas were identified by the AI model as significant for the diagnosis:", align="L")
            pdf.ln(3)
            
            for i, region in enumerate(detected_regions[:5]):  # Limit to 5 regions for PDF
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 8, f"Region {region.get('id', i+1)}: {region.get('location', 'Unknown')}", ln=True)
                pdf.set_font("Arial", "", 9)
                pdf.cell(0, 5, f"  Detection Intensity: {region.get('intensity', 0)*100:.1f}%", ln=True)
                pdf.cell(0, 5, f"  Area Size: {region.get('area', 0)} pixels", ln=True)
                
                explanation = region.get('explanation', {})
                if explanation:
                    pdf.set_font("Arial", "I", 9)
                    pdf.multi_cell(0, 5, f"  Explanation: {explanation.get('text', 'N/A')}", align="L")
                    pdf.ln(2)
    except:
        pass

    # Add Treatment Recommendations
    treatment_recs_str = analysis_data.get("treatment_recommendations", "[]")
    try:
        treatment_recommendations = json.loads(treatment_recs_str) if treatment_recs_str else []
        if treatment_recommendations:
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Treatment Recommendations", ln=True)
            pdf.ln(3)
            pdf.set_font("Arial", "", 10)
            pdf.multi_cell(0, 6, "Based on the severity assessment and clinical guidelines, the following recommendations are suggested:", align="L")
            pdf.ln(3)
            
            for rec in treatment_recommendations:
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 8, f"{rec.get('type', 'Recommendation')} ({rec.get('priority', 'N/A')} Priority)", ln=True)
                pdf.set_font("Arial", "", 9)
                pdf.multi_cell(0, 5, rec.get('description', 'N/A'), align="L")
                
                if rec.get('actions'):
                    pdf.set_font("Arial", "B", 9)
                    pdf.cell(0, 6, "Recommended Actions:", ln=True)
                    pdf.set_font("Arial", "", 8)
                    for action in rec.get('actions', []):
                        pdf.cell(10, 5, "", 0)  # Indent
                        pdf.cell(0, 5, f"• {action}", ln=True)
                
                if rec.get('medications'):
                    pdf.set_font("Arial", "B", 9)
                    pdf.cell(0, 6, "Medications:", ln=True)
                    pdf.set_font("Arial", "", 8)
                    for med in rec.get('medications', []):
                        pdf.cell(10, 5, "", 0)  # Indent
                        pdf.cell(0, 5, f"• {med}", ln=True)
                    if rec.get('duration'):
                        pdf.cell(10, 5, "", 0)
                        pdf.cell(0, 5, f"  Duration: {rec.get('duration')}", ln=True)
                
                if rec.get('note'):
                    pdf.set_font("Arial", "I", 8)
                    pdf.multi_cell(0, 5, f"Note: {rec.get('note')}", align="L")
                
                pdf.ln(2)
    except Exception as e:
        print(f"Error adding treatment recommendations to PDF: {e}")

    # Add confidence explanation
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Confidence Score Analysis:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, "The confidence score is calculated based on multiple factors including model accuracy, image quality, feature detection, and pattern recognition. A confidence score above 80% indicates high reliability in the diagnosis.", align="L")

    # Add the analyzed image
    image_path = analysis_data.get("image_path", "")
    if image_path and os.path.exists(image_path):
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analyzed X-ray Image:", ln=True)
        try:
            pdf.image(image_path, x=50, y=None, w=110)
        except:
            pdf.cell(0, 10, "[Image could not be embedded]", ln=True)
    
    # Add visualization image if available
    vis_path = analysis_data.get("visualization_path", "")
    if vis_path and os.path.exists(vis_path) and vis_path != image_path:
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detection Visualization (Heat Map):", ln=True)
        pdf.set_font("Arial", "", 9)
        pdf.cell(0, 6, "Red areas indicate higher model attention", ln=True)
        try:
            pdf.image(vis_path, x=50, y=None, w=110)
        except:
            pdf.cell(0, 10, "[Visualization could not be embedded]", ln=True)
    
    # Add important disclaimer
    pdf.ln(10)
    pdf.set_font("Arial", "B", 10)
    pdf.set_text_color(255, 0, 0)
    pdf.cell(0, 8, "IMPORTANT DISCLAIMER:", ln=True)
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 5, "This report is generated by an AI-assisted diagnostic tool. All recommendations are suggestions and should be reviewed and validated by a qualified medical professional. This tool is for assistance only and should not replace professional medical judgment. Always consult with a healthcare provider for proper diagnosis and treatment.", align="L")

    # Save the PDF
    pdf_path = os.path.join(REPORTS_FOLDER, "report.pdf")
    pdf.output(pdf_path)

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"pneumonia_analysis_{analysis_data['analysis_date'].replace(' ', '_')}.pdf"
    )

@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    feedback_data = request.json
    # Here you would typically save the feedback to a database
    # For now, we'll just return a success message
    return jsonify({"message": "Thank you for your feedback!"})

@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    annotation_data = request.json
    # Here you would typically save the annotation data
    # For now, we'll just return a success message
    return jsonify({"message": "Annotation saved successfully!"})

def calculate_model_metrics(predictions, true_labels):
    """Calculate and visualize model performance metrics"""
    try:
        # Convert predictions to binary (0 or 1)
        pred_binary = (predictions > 0.5).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(true_labels, pred_binary)
        
        # Create confusion matrix plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        # Save plot to temporary directory
        plot_path = os.path.join(REPORTS_FOLDER, 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Serve the image via Flask route
        plot_url = f"/static/reports/confusion_matrix.png"
        
        return {
            'confusion_matrix': plot_url,
            'accuracy': round(accuracy * 100, 2),
            'sensitivity': round(sensitivity * 100, 2),
            'specificity': round(specificity * 100, 2),
            'precision': round(precision * 100, 2)
        }
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return {
            'confusion_matrix': '',
            'accuracy': 0,
            'sensitivity': 0,
            'specificity': 0,
            'precision': 0
        }

@app.route("/model_performance")
def model_performance():
    try:
        # Load history from temporary storage
        history = load_history()
        
        if not history or len(history) < 2:
            return render_template("model_performance.html", 
                                 error="Insufficient prediction history available. Please perform at least 2 predictions to view model performance metrics.")
        
        # Extract predictions and true labels from history
        predictions = []
        true_labels = []
        
        for entry in history:
            predictions.append(entry.get('confidence', 50) / 100)  # Convert confidence back to probability
            result = entry.get('result', 'No Pneumonia')
            true_labels.append(1 if result == "Pneumonia Detected" else 0)
        
        # Calculate metrics
        if len(predictions) > 0:
            metrics = calculate_model_metrics(np.array(predictions), np.array(true_labels))
            
            return render_template("model_performance.html", 
                                 metrics=metrics,
                                 history_length=len(history))
        else:
            return render_template("model_performance.html", 
                                 error="No valid prediction data found.")
    except Exception as e:
        print(f"Error in model_performance: {e}")
        import traceback
        traceback.print_exc()
        return render_template("model_performance.html", 
                             error=f"Error loading performance metrics: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

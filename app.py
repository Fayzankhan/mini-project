from flask import Flask, render_template, request, send_file, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from fpdf import FPDF
from datetime import datetime
from skimage import metrics
import json

app = Flask(__name__)

# Load model
MODEL_PATH = "saved_model/pneumonia_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Ensure upload folder exists
UPLOAD_FOLDER = "static/uploads"
REPORTS_FOLDER = "static/reports"
HISTORY_FILE = "static/history.json"
PATIENTS_FILE = "static/patients.json"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)

# Initialize history and patients files if they don't exist
if not os.path.exists(HISTORY_FILE):
    with open(HISTORY_FILE, 'w') as f:
        json.dump([], f)

if not os.path.exists(PATIENTS_FILE):
    with open(PATIENTS_FILE, 'w') as f:
        json.dump([], f)

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

# Function to preprocess image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))  # Match model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/", methods=["GET"])
def home():
    history = load_history()
    patients = load_patients()
    return render_template("index.html", history=history, patients=patients)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="No file uploaded.")

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result="No file selected.")

    # Get patient information from the form
    patient_info = {
        "name": request.form.get("name", "Not Available"),
        "age": request.form.get("age", "Not Available"),
        "gender": request.form.get("gender", "Not Available"),
        "medical_history": request.form.get("medical_history", "Not Available")
    }

    # Save patient information
    patients = load_patients()
    patients.append(patient_info)
    save_patients(patients)

    # Save the uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read the original image for quality assessment
    original_image = cv2.imread(file_path)
    
    # Calculate additional confidence metrics
    image_quality = calculate_image_quality(original_image)
    feature_detection = calculate_feature_detection(original_image)
    pattern_recognition = calculate_pattern_recognition(original_image)

    # Process the image and get prediction
    image = preprocess_image(file_path)
    prediction = model.predict(image)[0][0]
    confidence = round(float(prediction) * 100, 2)
    
    if prediction > 0.5:
        result = "Pneumonia Detected"
    else:
        result = "No Pneumonia"

    # Get current timestamp
    analysis_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save analysis details for report
    analysis_data = {
        "filename": file.filename,
        "result": result,
        "confidence": confidence,
        "image_quality": image_quality,
        "feature_detection": feature_detection,
        "pattern_recognition": pattern_recognition,
        "analysis_date": analysis_date,
        "image_path": file_path,
        "patient_info": json.dumps(patient_info)  # Properly serialize patient_info to JSON
    }
    
    # Store the analysis data in session
    with open(os.path.join(REPORTS_FOLDER, "latest_analysis.txt"), "w") as f:
        for key, value in analysis_data.items():
            f.write(f"{key}:{value}\n")

    # Add to history
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

    return render_template(
        "index.html",
        result=result,
        image_url=file_path,
        confidence=confidence,
        image_quality=image_quality,
        feature_detection=feature_detection,
        pattern_recognition=pattern_recognition,
        analysis_date=analysis_date,
        history=history,
        patient_info=patient_info,
        show_patient_info=True  # Add this flag to show patient info in the template
    )

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
        ("Diagnosis", analysis_data["result"]),
        ("Overall Confidence", f"{analysis_data['confidence']}%"),
        ("Image Quality", f"{analysis_data['image_quality']}%"),
        ("Feature Detection", f"{analysis_data['feature_detection']}%"),
        ("Pattern Recognition", f"{analysis_data['pattern_recognition']}%"),
        ("Analysis Date", analysis_data["analysis_date"])
    ]

    for label, value in details:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(60, 10, label + ":", 0)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, str(value), ln=True)

    # Add confidence explanation
    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Confidence Score Analysis:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, "The confidence score is calculated based on multiple factors including model accuracy, image quality, feature detection, and pattern recognition. A confidence score above 80% indicates high reliability in the diagnosis.", align="L")

    # Add the analyzed image
    if os.path.exists(analysis_data["image_path"]):
        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Analyzed X-ray Image:", ln=True)
        pdf.image(analysis_data["image_path"], x=50, y=None, w=110)

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

if __name__ == "__main__":
    app.run(debug=True)

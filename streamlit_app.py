"""
Streamlit Frontend for Pneumonia Detection
Backend API runs on Render
"""

import streamlit as st
import requests
import json
from PIL import Image
import io
import os

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0f3460 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #0f3460 100%);
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        background: rgba(28, 28, 45, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .severity-high {
        background: rgba(255, 107, 107, 0.2);
        border-left: 4px solid #ff6b6b;
    }
    .severity-moderate {
        background: rgba(255, 165, 0, 0.2);
        border-left: 4px solid #ffa500;
    }
    .severity-mild {
        background: rgba(78, 205, 196, 0.2);
        border-left: 4px solid #4ecdc4;
    }
    </style>
""", unsafe_allow_html=True)

# Backend API URL (Render deployment)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8080")

def predict_pneumonia(image_file, patient_info):
    """Send prediction request to Flask backend"""
    try:
        files = {'file': image_file}
        data = patient_info
        
        response = requests.post(
            f"{BACKEND_URL}/predict",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.text, None
        else:
            return None, f"Error: {response.status_code}"
    except Exception as e:
        return None, str(e)

def main():
    st.title("🫁 Pneumonia Detection System")
    st.markdown("---")
    
    # Sidebar for patient information
    with st.sidebar:
        st.header("Patient Information")
        patient_name = st.text_input("Full Name", key="name")
        patient_age = st.number_input("Age", min_value=0, max_value=120, value=30, key="age")
        patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
        medical_history = st.text_area("Medical History", key="history")
        
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("""
        1. Fill in patient information
        2. Upload a chest X-ray image
        3. Click 'Analyze' to get results
        4. Review severity assessment and recommendations
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
    
    with col2:
        st.header("Analysis Results")
        
        if uploaded_file is not None:
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing X-ray image... This may take a moment."):
                    # Prepare patient info
                    patient_info = {
                        'name': patient_name or 'Not Provided',
                        'age': str(patient_age),
                        'gender': patient_gender.lower(),
                        'medical_history': medical_history or 'None'
                    }
                    
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Call backend API
                    result_html, error = predict_pneumonia(uploaded_file, patient_info)
                    
                    if error:
                        st.error(f"Error: {error}")
                        st.info("Make sure the backend API is running on Render")
                    elif result_html:
                        # Display results (simplified - in production, parse JSON response)
                        st.success("Analysis Complete!")
                        st.markdown("---")
                        
                        # Note: For full integration, backend should return JSON
                        # For now, showing a message
                        st.info("""
                        **Analysis Results:**
                        - The backend API has processed your image
                        - Full results are available on the Flask backend
                        - Check the Render deployment URL for complete analysis
                        """)
        else:
            st.info("👆 Please upload an X-ray image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Powered by TensorFlow & Flask | Backend API: {}</p>
        <p><small>This tool is for assistance only. Always consult medical professionals.</small></p>
    </div>
    """.format(BACKEND_URL), unsafe_allow_html=True)

if __name__ == "__main__":
    main()


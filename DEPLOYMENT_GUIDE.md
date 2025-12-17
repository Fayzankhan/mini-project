# Deployment Guide: Flask on Render + Streamlit Option

## 🚀 Option 1: Deploy Flask App on Render (Recommended)

### Step 1: Prepare Your Repository

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

### Step 2: Deploy on Render

1. **Go to Render Dashboard**
   - Visit: https://render.com
   - Sign up/Login

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

3. **Configure Settings**
   - **Name**: `pneumonia-detection-app`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: Free tier (or paid for better performance)

4. **Environment Variables** (if needed)
   - `PYTHON_VERSION`: `3.10.12`
   - `PORT`: `10000` (Render sets this automatically)

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (~5-10 minutes)
   - Your app will be live at: `https://pneumonia-detection-app.onrender.com`

### Step 3: Update Static Files Path

The app should work, but you may need to ensure static files are served correctly.

---

## 🎨 Option 2: Streamlit Frontend + Flask Backend on Render

### Architecture:
- **Frontend**: Streamlit (deployed on Streamlit Cloud)
- **Backend**: Flask API (deployed on Render)

### Step 1: Deploy Flask Backend API on Render

1. **Create API-only Flask app** (`api_app.py`):
   ```python
   # Similar to app.py but returns JSON instead of HTML
   # Add /api/predict endpoint that returns JSON
   ```

2. **Deploy on Render** (same as Option 1)

3. **Get your Render URL**: `https://your-app.onrender.com`

### Step 2: Deploy Streamlit Frontend

1. **Update `streamlit_app.py`** with your Render backend URL:
   ```python
   BACKEND_URL = "https://your-app.onrender.com"
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to: https://streamlit.io/cloud
   - Connect GitHub repository
   - Select `streamlit_app.py` as main file
   - Set environment variable: `BACKEND_URL=https://your-app.onrender.com`
   - Deploy!

---

## 📋 Files Created for Deployment

### For Render Deployment:
- ✅ `render.yaml` - Render configuration
- ✅ `runtime.txt` - Python version
- ✅ `Procfile` - Already exists (web: gunicorn app:app)
- ✅ `requirements.txt` - Dependencies

### For Streamlit Option:
- ✅ `streamlit_app.py` - Streamlit frontend
- ✅ `.streamlit/config.toml` - Streamlit config

---

## 🔧 Required Changes for Render

### 1. Update app.py for Production

Add this to handle static files properly:

```python
# At the top of app.py
import os
from flask import send_from_directory

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)
```

### 2. Handle File Storage

For Render, you might want to use cloud storage (S3, etc.) instead of temp files:

```python
# Option: Use environment variable for storage
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', os.path.join(tempfile.gettempdir(), "uploads"))
```

### 3. Model File Size

- Render free tier has limits
- Consider using model hosting service or CDN for large model files
- Or use Render paid tier

---

## 🌐 Environment Variables for Render

Set these in Render dashboard:

```
PYTHON_VERSION=3.10.12
PORT=10000
FLASK_ENV=production
BACKEND_URL=https://your-app.onrender.com  # If using Streamlit
```

---

## 📦 Build Optimization

### Reduce Build Time:

1. **Use lighter TensorFlow**:
   ```txt
   tensorflow-cpu==2.8.0  # Already using CPU version ✅
   ```

2. **Add .dockerignore** (if using Docker):
   ```
   __pycache__/
   *.pyc
   .git/
   dataset/
   *.h5
   ```

3. **Pre-download models** or use model hosting

---

## 🚨 Important Notes

### Render Free Tier Limitations:
- ⚠️ Spins down after 15 minutes of inactivity
- ⚠️ Limited build time (45 minutes)
- ⚠️ Limited memory (512MB)
- ⚠️ Large model files may cause issues

### Solutions:
1. **Use Render Paid Tier** ($7/month) for better performance
2. **Host models separately** (AWS S3, Google Cloud Storage)
3. **Use model optimization** (quantization, pruning)
4. **Consider alternative**: Railway.app, Fly.io, or Heroku

---

## 🔄 Alternative: Railway.app Deployment

Railway is another good option:

1. Go to: https://railway.app
2. Connect GitHub
3. Deploy automatically
4. Better free tier than Render

---

## 📝 Quick Deploy Checklist

- [ ] Code pushed to GitHub
- [ ] `requirements.txt` updated
- [ ] `Procfile` exists
- [ ] `runtime.txt` exists
- [ ] Model files accessible
- [ ] Environment variables set
- [ ] Static files configured
- [ ] Test locally first

---

## 🆘 Troubleshooting

### Build Fails:
- Check `requirements.txt` for compatibility
- Ensure Python version matches
- Check build logs in Render dashboard

### App Crashes:
- Check logs in Render dashboard
- Verify model file paths
- Check file permissions

### Slow Performance:
- Upgrade to paid tier
- Optimize model size
- Use CDN for static files

---

## 📞 Support

- Render Docs: https://render.com/docs
- Streamlit Cloud: https://docs.streamlit.io/streamlit-cloud
- Flask Deployment: https://flask.palletsprojects.com/en/2.3.x/deploying/

---

**Recommended**: Start with Option 1 (Full Flask on Render) as it's simpler and your app is already built for Flask!


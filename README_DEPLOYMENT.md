# Quick Deployment Guide

## 🚀 Deploy Flask App on Render (Easiest)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Ready for deployment"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to https://render.com and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `pneumonia-detection-app`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click "Create Web Service"
6. Wait ~5-10 minutes for deployment
7. Your app is live! 🎉

### Your App URL:
`https://pneumonia-detection-app.onrender.com`

---

## 🎨 Option: Streamlit Frontend + Flask Backend

### Backend (Render):
1. Deploy Flask app on Render (same as above)
2. Get your Render URL: `https://your-app.onrender.com`

### Frontend (Streamlit Cloud):
1. Update `streamlit_app.py` line 15:
   ```python
   BACKEND_URL = "https://your-app.onrender.com"
   ```
2. Go to https://streamlit.io/cloud
3. Connect GitHub repo
4. Select `streamlit_app.py` as main file
5. Add environment variable: `BACKEND_URL=https://your-app.onrender.com`
6. Deploy!

---

## ⚠️ Important Notes

1. **Model Files**: Make sure your model files are in the repo or hosted elsewhere
2. **Free Tier**: Render free tier spins down after 15 min inactivity
3. **Build Time**: First build may take 10-15 minutes (TensorFlow is large)
4. **Memory**: Free tier has 512MB RAM - may need paid tier for large models

---

## 📋 Files Created

- ✅ `render.yaml` - Render config
- ✅ `runtime.txt` - Python version
- ✅ `Procfile` - Start command
- ✅ `streamlit_app.py` - Streamlit frontend (optional)
- ✅ `.gitignore` - Git ignore rules
- ✅ `DEPLOYMENT_GUIDE.md` - Detailed guide

---

## 🔧 Troubleshooting

**Build fails?**
- Check Render build logs
- Ensure all dependencies in requirements.txt
- Python version matches runtime.txt

**App crashes?**
- Check Render logs
- Verify model file paths
- Check file permissions

**Slow?**
- Upgrade to Render paid tier ($7/month)
- Optimize model size
- Use CDN for static files

---

**Ready to deploy?** Follow Step 1 & 2 above! 🚀



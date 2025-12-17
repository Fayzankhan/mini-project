# Render Deployment Steps - Quick Reference

## Your GitHub Repository
- **URL**: https://github.com/Fayzankhan/mini-project.git
- **Status**: ✅ Code is pushed

## Render Deployment Checklist

### ✅ Step 1: Go to Render
1. Visit: https://render.com
2. Sign up/Login (use GitHub for easy connection)

### ✅ Step 2: Create Web Service
1. Click **"New +"** button (top right)
2. Select **"Web Service"**
3. Connect GitHub if not already connected
4. Select repository: **`Fayzankhan/mini-project`**

### ✅ Step 3: Configure Service

**Basic Settings:**
- **Name**: `pneumonia-detection-app`
- **Environment**: `Python 3`
- **Region**: Choose closest to you
- **Branch**: `main` (or your default branch)

**Build & Deploy:**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn app:app`

**Plan:**
- Start with **Free** (can upgrade later)

### ✅ Step 4: Deploy
1. Click **"Create Web Service"**
2. Wait for build (10-15 minutes first time)
3. Watch build logs for progress

### ✅ Step 5: Access Your App
- Your app URL: `https://pneumonia-detection-app.onrender.com`
- (Or whatever name you chose)

## Important Notes

### ⚠️ First Build
- Takes 10-15 minutes (TensorFlow installation)
- Be patient, it's normal!

### ⚠️ Free Tier Limitations
- Spins down after 15 min inactivity
- First request after spin-down takes ~30 seconds
- 512MB RAM limit

### ⚠️ Model Files
- Make sure your model files are in the repo
- Or host them separately (S3, etc.)

## Troubleshooting

### Build Fails?
1. Check build logs in Render dashboard
2. Common issues:
   - Missing dependencies in requirements.txt
   - Python version mismatch
   - Model file too large

### App Crashes?
1. Check logs in Render dashboard
2. Common issues:
   - Model file path incorrect
   - Missing directories
   - Memory limit exceeded

### Slow Performance?
- Upgrade to Starter plan ($7/month)
- Optimize model size
- Use CDN for static files

## After Deployment

1. **Test your app**: Visit the URL
2. **Check logs**: Monitor for errors
3. **Update if needed**: Push to GitHub, Render auto-deploys

## Your Deployment URL
Once deployed, your app will be live at:
**https://pneumonia-detection-app.onrender.com**

(Or your custom name)

---

**Need help?** Check Render docs: https://render.com/docs



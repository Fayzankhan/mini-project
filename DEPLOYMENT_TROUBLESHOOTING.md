# Deployment Troubleshooting Guide

## Issue: "Preparing metadata" taking too long

### Quick Fixes:

#### Option 1: Optimize requirements.txt (Already Done ✅)
- Removed optional dependencies (shap, lime) - can add later
- Using tensorflow-cpu (lighter)
- Added build optimizations

#### Option 2: Use Pre-built Wheels
The updated `render.yaml` now includes:
- `--no-cache-dir` flag
- Upgraded pip/setuptools/wheel first
- This helps use pre-built wheels instead of building from source

#### Option 3: Remove Heavy Dependencies Temporarily
If still slow, temporarily remove:
- matplotlib
- seaborn
- shap
- lime

You can add them back after deployment.

### Alternative: Use Lighter TensorFlow

If TensorFlow is the bottleneck, you can:

1. **Use TensorFlow Lite** (much smaller):
   ```txt
   tensorflow-lite==2.8.0
   ```

2. **Or skip TensorFlow in requirements**, load model differently

### Build Time Optimization

The updated configuration:
- ✅ Upgrades pip first (faster dependency resolution)
- ✅ Uses `--no-cache-dir` (saves space, sometimes faster)
- ✅ Disables pip version check
- ✅ Sets timeout to prevent hanging

### Expected Build Times:
- **First build**: 10-20 minutes (normal for TensorFlow)
- **Subsequent builds**: 5-10 minutes (cached dependencies)

### If Still Too Slow:

1. **Cancel and retry** - Sometimes network issues cause delays
2. **Check Render status** - https://status.render.com
3. **Use Render paid tier** - Faster build servers
4. **Split dependencies** - Install core deps first, optional later

### Monitor Build Logs

Watch for:
- ✅ "Collecting..." - Normal, downloading packages
- ✅ "Building wheels..." - Normal, compiling if needed
- ⚠️ "ERROR" - Something failed
- ⚠️ Hanging at one package - Network issue, cancel and retry

### Quick Test

After deployment, test with:
```bash
curl https://your-app.onrender.com
```

---

**Current Status**: Requirements optimized, build config updated ✅


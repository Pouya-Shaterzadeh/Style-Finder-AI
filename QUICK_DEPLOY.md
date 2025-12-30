# Quick Deployment to Hugging Face Spaces

## 🚀 Fast Track (5 Minutes)

### Step 1: Create Space
1. Go to https://huggingface.co/spaces
2. Click **"Create new Space"**
3. Fill in:
   - **Name**: `style-finder-ai`
   - **SDK**: `Docker`
   - **Hardware**: `CPU basic`
4. Click **"Create Space"**

### Step 2: Push Code
```bash
cd /home/pouya/Style-Finder-AI
git add .
git commit -m "Deploy to Hugging Face Spaces"
git remote add origin https://huggingface.co/spaces/PouyaDevA1/style-finder-ai
git push origin main
```

### Step 3: Wait & Enjoy
- Build takes 10-15 minutes (first time)
- Your app: https://huggingface.co/spaces/PouyaDevA1/style-finder-ai

## 📋 What's Included

✅ Dockerfile with Chrome/Selenium support  
✅ All dependencies in requirements.txt  
✅ README.md for Space description  
✅ Optimized for Hugging Face Spaces  
✅ Automatic model download (BLIP ~1GB)  

## ⚙️ Optional: Add API Token

1. Space → Settings → Variables and secrets
2. Add secret: `HF_API_TOKEN` = your token
3. (Not required, but improves rate limits)

## 📖 Full Guide

See `DEPLOYMENT.md` for detailed instructions.


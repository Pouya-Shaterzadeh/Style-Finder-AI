# Deployment Guide for Hugging Face Spaces

This guide will help you deploy Style Finder AI to your Hugging Face Spaces profile.

## Prerequisites

- A Hugging Face account (create one at https://huggingface.co/join if needed)
- Your Hugging Face username: `PouyaDevA1`

## Step-by-Step Deployment

### 1. Create a New Space

1. Go to https://huggingface.co/spaces
2. Click the **"Create new Space"** button
3. Fill in the Space creation form:
   - **Space name**: `style-finder-ai` (or your preferred name)
   - **SDK**: Select **"Docker"**
   - **Hardware**: Select **"CPU basic"** (or GPU if you have access)
   - **Visibility**: Choose **"Public"** (or Private if you prefer)
4. Click **"Create Space"**

### 2. Upload Your Code

You have two options:

#### Option A: Using Git (Recommended)

1. **Initialize Git in your project** (if not already done):
   ```bash
   cd /home/pouya/Style-Finder-AI
   git init
   git add .
   git commit -m "Initial commit for Hugging Face Spaces deployment"
   ```

2. **Add your Hugging Face Space as a remote**:
   ```bash
   git remote add origin https://huggingface.co/spaces/PouyaDevA1/style-finder-ai
   ```

3. **Push your code**:
   ```bash
   git push origin main
   ```
   (If your default branch is `master`, use `git push origin master`)

#### Option B: Using Hugging Face Web Interface

1. Go to your Space page: `https://huggingface.co/spaces/PouyaDevA1/style-finder-ai`
2. Click **"Files and versions"** tab
3. Click **"Add file"** → **"Upload files"**
4. Upload all necessary files:
   - `app.py`
   - `Dockerfile`
   - `requirements.txt`
   - `README.md`
   - `config/` directory (all files)
   - `src/` directory (all files)
   - `static/` directory (all files)

### 3. Set Environment Variables (Optional)

1. Go to your Space page
2. Click **"Settings"** tab
3. Scroll to **"Variables and secrets"** section
4. Click **"Add secret"**
5. Add your Hugging Face API token (optional, for enhanced access):
   - **Name**: `HF_API_TOKEN`
   - **Value**: Your Hugging Face API token
   - Click **"Add secret"**

> **Note**: The app works without a token, but having one provides better rate limits.

### 4. Wait for Build

1. After pushing/uploading files, Hugging Face will automatically start building your Space
2. You can monitor the build progress in the **"Logs"** tab
3. First build typically takes **10-15 minutes** because:
   - Docker image is being built
   - Python dependencies are installed
   - BLIP model (~1GB) downloads automatically
   - Chrome and ChromeDriver are installed for Selenium

### 5. Access Your App

Once the build completes successfully:
- Your app will be available at: `https://huggingface.co/spaces/PouyaDevA1/style-finder-ai`
- The app will automatically reload when you push new changes

## File Structure for Deployment

Ensure these files are present in your Space:

```
style-finder-ai/
├── app.py                 # Main Gradio application
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies
├── README.md              # Space description (shown on Space page)
├── .gitignore             # Git ignore rules
├── config/
│   └── config.py         # Configuration settings
├── src/
│   ├── vlm_service.py     # VLM model wrapper
│   ├── trendyol_scraper.py # Trendyol scraper
│   ├── fashion_analyzer.py # Main analysis pipeline
│   ├── image_processor.py  # Image preprocessing
│   └── utils.py           # Helper functions
└── static/
    └── css/
        └── custom.css     # Custom UI styling
```

## Important Notes

### Model Download
- The BLIP model downloads automatically on first Space start
- Model size: ~1GB
- Download happens once, then stays cached

### Build Time
- First build: 10-15 minutes
- Subsequent builds: 3-5 minutes (faster due to caching)

### Hardware Requirements
- **CPU basic**: Sufficient for the app (recommended)
- **GPU**: Optional, speeds up inference but not required

### Selenium Setup
- Chrome and ChromeDriver are automatically installed in the Dockerfile
- Selenium runs in headless mode (no display needed)
- If Selenium fails, the app falls back to demo product links

## Troubleshooting

### Build Fails

1. **Check Logs**: Go to Space → Logs tab to see error messages
2. **Common Issues**:
   - Syntax errors in Dockerfile
   - Missing dependencies in requirements.txt
   - File path issues

### App Doesn't Start

1. **Check Logs**: Look for Python errors in the Logs tab
2. **Common Issues**:
   - Missing environment variables
   - Model download failures
   - Port conflicts (should use 7860)

### Model Loading Issues

1. **First Request**: First request may take 30-60 seconds (model loading)
2. **Subsequent Requests**: Should be faster
3. **Memory Issues**: Consider upgrading to GPU hardware if available

### Selenium Errors

- The app has fallback mechanisms
- If Selenium fails, demo product links are used
- Check Logs for specific Selenium errors

## Updating Your Space

To update your Space with new code:

1. **Make changes locally**
2. **Commit changes**:
   ```bash
   git add .
   git commit -m "Update description"
   ```
3. **Push to Space**:
   ```bash
   git push origin main
   ```
4. **Wait for rebuild** (automatic)

## Monitoring

- **Logs**: Check the Logs tab for runtime errors
- **Metrics**: View usage statistics in the Space dashboard
- **Settings**: Adjust hardware, visibility, and other settings

## Support

If you encounter issues:
1. Check the Logs tab for error messages
2. Review the README.md for configuration options
3. Check Hugging Face Spaces documentation: https://huggingface.co/docs/hub/spaces

## Your Space URL

Once deployed, your app will be available at:
**https://huggingface.co/spaces/PouyaDevA1/style-finder-ai**

Enjoy your deployed fashion analysis app! 🎉


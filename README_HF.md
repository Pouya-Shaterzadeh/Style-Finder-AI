# Hugging Face Spaces Deployment Guide

This document provides instructions for deploying Style Finder AI to Hugging Face Spaces.

## Quick Deployment Steps

1. **Create a New Space**:
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Fill in the details:
     - **Space name**: `style-finder-ai` (or your preferred name)
     - **SDK**: Select "Docker"
     - **Hardware**: Select "CPU basic" (or GPU if available)
     - **Visibility**: Public or Private
   - Click "Create Space"

2. **Upload Your Code**:
   - Clone your repository or upload files directly
   - Ensure all files are present:
     - `app.py`
     - `Dockerfile`
     - `requirements.txt`
     - `README.md` (this will be the Space description)
     - `config/` directory
     - `src/` directory
     - `static/` directory

3. **Set Environment Variables** (Optional):
   - Go to Space → Settings → Variables and secrets
   - Add a new Secret:
     - **Name**: `HF_API_TOKEN`
     - **Value**: Your Hugging Face API token (optional, for enhanced access)
   - Click "Add secret"

4. **Wait for Build**:
   - The Space will automatically build using the Dockerfile
   - First build may take 10-15 minutes (model download)
   - Subsequent builds are faster

5. **Access Your App**:
   - Once built, your app will be available at:
     `https://huggingface.co/spaces/PouyaDevA1/style-finder-ai`

## Important Notes

- **Model Download**: The BLIP model (~1GB) downloads automatically on first Space start
- **Build Time**: First build takes longer due to model and dependency installation
- **Hardware**: CPU basic is sufficient, but GPU speeds up inference
- **Selenium**: Chrome and ChromeDriver are installed in the Dockerfile for web scraping

## Troubleshooting

- **Build Fails**: Check Dockerfile syntax and ensure all dependencies are in requirements.txt
- **Model Loading Issues**: Ensure transformers and torch are in requirements.txt
- **Selenium Errors**: Chrome/ChromeDriver are installed in Dockerfile - should work automatically
- **Memory Issues**: Consider upgrading to GPU hardware if available

## Space Configuration

The Space uses:
- **SDK**: Docker
- **Python Version**: 3.11
- **Port**: 7860 (Gradio default)
- **Model**: BLIP (Salesforce/blip-image-captioning-base)

For more details, see the main README.md file.


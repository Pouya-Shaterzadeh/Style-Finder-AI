#!/bin/bash
# Script to initialize Git repository and prepare for GitHub

echo "Initializing Git repository for Style Finder AI..."

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized."
fi

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Style Finder AI - Computer Vision-Based Fashion Analysis

- Implemented LLaVA VLM service for fashion image analysis
- Integrated Trendyol.com scraper for product search
- Built multi-stage matching algorithm with visual similarity
- Created rich UI/UX with Gradio and custom CSS
- Prepared Dockerfile for Hugging Face Spaces deployment"

echo ""
echo "Repository initialized and initial commit created!"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub (e.g., 'style-finder-ai')"
echo "2. Add the remote: git remote add origin https://github.com/YOUR_USERNAME/style-finder-ai.git"
echo "3. Push to GitHub: git push -u origin main"
echo ""
echo "For Hugging Face Spaces deployment:"
echo "1. Go to https://huggingface.co/spaces/PouyaDevA1"
echo "2. Create a new Space"
echo "3. Select 'Docker' as the SDK"
echo "4. Push this repository to the Space"


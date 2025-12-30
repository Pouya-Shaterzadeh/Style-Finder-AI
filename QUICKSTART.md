# Quick Start Guide - Style Finder AI

## Local Development Setup

### 1. Prerequisites
- Python 3.11 or higher
- Git
- 8GB+ RAM (for model loading)
- GPU recommended but not required

### 2. Clone and Setup

```bash
# Navigate to project directory
cd Style-Finder-AI

# Run setup script
./setup.sh

# Or manually:
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Application

```bash
# Activate virtual environment (if not already active)
source venv/bin/activate

# Run the app
python app.py
```

The application will start on `http://localhost:7860`

## GitHub Repository Setup

### 1. Initialize Repository

```bash
# Run the initialization script
./init_repo.sh

# Or manually:
git init
git add .
git commit -m "Initial commit: Style Finder AI"
```

### 2. Create GitHub Repository

1. Go to GitHub and create a new repository (e.g., `style-finder-ai`)
2. Add remote and push:

```bash
git remote add origin https://github.com/YOUR_USERNAME/style-finder-ai.git
git branch -M main
git push -u origin main
```

## Hugging Face Spaces Deployment

### 1. Create a Space

1. Go to https://huggingface.co/spaces/PouyaDevA1
2. Click "Create new Space"
3. Fill in:
   - **Space name**: `style-finder-ai`
   - **SDK**: `Docker`
   - **Visibility**: `Public`
4. Click "Create Space"

### 2. Push to Hugging Face

```bash
# Install Hugging Face CLI (if not installed)
pip install huggingface_hub[cli]

# Login to Hugging Face
huggingface-cli login

# Clone your Space (replace with your space name)
git clone https://huggingface.co/spaces/PouyaDevA1/style-finder-ai
cd style-finder-ai

# Copy all files from your project
cp -r /path/to/Style-Finder-AI/* .

# Commit and push
git add .
git commit -m "Deploy Style Finder AI"
git push
```

### 3. Alternative: Direct Push from Local

```bash
# Add Hugging Face as remote
git remote add hf https://huggingface.co/spaces/PouyaDevA1/style-finder-ai

# Push to Hugging Face
git push hf main
```

## Project Structure

```
Style-Finder-AI/
├── app.py                 # Main Gradio application
├── src/
│   ├── vlm_service.py     # LLaVA model integration
│   ├── trendyol_scraper.py # Trendyol.com scraper
│   ├── image_processor.py  # Image processing & CLIP
│   ├── fashion_analyzer.py # Main analysis pipeline
│   └── utils.py           # Utility functions
├── config/
│   └── config.py         # Configuration settings
├── static/
│   ├── css/
│   │   └── custom.css    # Custom UI styling
│   └── images/           # Example images
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── README.md            # Project documentation
└── README_HF.md        # Hugging Face Space README
```

## Configuration

Edit `config/config.py` to adjust:
- Model names and settings
- Trendyol search parameters
- Similarity thresholds
- UI settings

## Troubleshooting

### Model Loading Issues
- Ensure you have enough RAM (8GB+ recommended)
- First run will download models (~7GB for LLaVA-7B)
- Use CPU if GPU is not available (slower but works)

### Trendyol Scraping Issues
- Rate limiting: Increase `REQUEST_DELAY` in config
- HTML structure changes: Update selectors in `trendyol_scraper.py`
- Network errors: Check internet connection

### Import Errors
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again
- Check Python version (3.11+)

## Performance Tips

1. **First Run**: Models will be downloaded (~7GB), be patient
2. **GPU**: Use GPU for faster inference (automatic if available)
3. **Caching**: VLM responses are not cached by default (can be added)
4. **Image Size**: Larger images take longer to process

## Next Steps

- Test with various fashion images
- Monitor accuracy and adjust similarity thresholds
- Add Turkish language support (future update)
- Expand product database or add more e-commerce sites

## Support

For issues or questions, check:
- README.md for detailed documentation
- GitHub Issues for known problems
- Hugging Face Space discussions


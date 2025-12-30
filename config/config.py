"""
Configuration settings for Style Finder AI application
"""
import os
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ Loaded environment variables from {env_path}")
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

# Model Configuration
# 
# DEPLOYMENT OPTIONS:
# 1. Hugging Face Spaces (RECOMMENDED) - Model loads on HF servers, users just use the web app
# 2. Local development - Model downloads to your machine on first run
#
# The Inference API is NOT available for most vision models, so we use local loading.
# On HF Spaces, the model downloads ONCE when the Space starts, then stays loaded.

# Use local model loading (works on HF Spaces and local development)
USE_INFERENCE_API = False  # Set to False to use local models (recommended)

# Vision Language Model - choose based on your needs:
# Option 1: BLIP (smaller, faster, ~1GB) - Good for basic captioning
LLAVA_MODEL_NAME = "Salesforce/blip-image-captioning-base"

# Option 2: Moondream (small VLM, ~2GB) - Better quality, still lightweight
# LLAVA_MODEL_NAME = "vikhyatk/moondream2"

# Option 3: LLaVA (best quality, ~7GB) - Requires more resources
# LLAVA_MODEL_NAME = "llava-hf/llava-1.5-7b-hf"

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

# Hugging Face API Configuration (for token - still needed for some operations)
HF_API_URL = "https://router.huggingface.co/inference"  # Router endpoint (mostly unavailable)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")  # Optional - for API access if available

# Debug: Check if token is loaded (without exposing the full token)
if HF_API_TOKEN:
    print(f"✓ Hugging Face API token loaded (starts with: {HF_API_TOKEN[:10]}...)")
else:
    print("⚠️  Warning: HF_API_TOKEN not found. The app will work but with lower rate limits.")

# Image Processing
IMAGE_SIZE = (512, 512)
MAX_IMAGE_SIZE = 1024

# Fashion Analysis
MAX_CLOTHING_ITEMS = 10
CONFIDENCE_THRESHOLD = 0.3

# Trendyol Configuration
TRENDYOL_BASE_URL = "https://www.trendyol.com"
TRENDYOL_SEARCH_URL = "https://www.trendyol.com/sr"
MAX_SEARCH_RESULTS = 20
REQUEST_DELAY = 1.0  # Delay between requests in seconds

# Matching Configuration
VISUAL_SIMILARITY_WEIGHT = 0.6
TEXT_SIMILARITY_WEIGHT = 0.4
MIN_SIMILARITY_SCORE = 0.3  # Lowered to show more results, especially demo products

# UI Configuration
GRADIO_THEME = "soft"
MAX_RESULTS_DISPLAY = 12

# Device Configuration
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"


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
    pass

# ---------------------------------------------------------------------------
# VLM Configuration
# Using Qwen2-VL-7B-Instruct via HF Serverless Inference API
# - Open access (no gated model approval required)
# - True multimodal LLM with visual reasoning
# - Returns structured JSON from a single prompt
# ---------------------------------------------------------------------------
VLM_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
LLAVA_MODEL_NAME = VLM_MODEL_NAME  # Keep alias for backward compat

# Always use Server-side Inference API (no local model download on HF Spaces)
USE_INFERENCE_API = True

# ---------------------------------------------------------------------------
# Visual Similarity - Fashion-specific CLIP
# patrickjohncyh/fashion-clip: trained on 800K+ fashion image-text pairs
# Drop-in replacement for openai/clip-vit-base-patch32 (same API)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"

# ---------------------------------------------------------------------------
# Hugging Face API Configuration
# ---------------------------------------------------------------------------
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

if HF_API_TOKEN:
    print(f"✓ Hugging Face API token loaded (starts with: {HF_API_TOKEN[:10]}...)")
else:
    print("⚠️  Warning: HF_API_TOKEN not found. Set this env var for VLM access.")

# ---------------------------------------------------------------------------
# Trendyol Configuration
# Using internal JSON API instead of HTML scraping (no bot detection)
# ---------------------------------------------------------------------------
TRENDYOL_BASE_URL = "https://www.trendyol.com"
TRENDYOL_SEARCH_URL = "https://www.trendyol.com/sr"
TRENDYOL_JSON_API_URL = (
    "https://public.trendyol.com/discovery-web-searchgw-service/api/"
    "infinite-scroll/product-search-with-typed-items"
)
MAX_SEARCH_RESULTS = 20
REQUEST_DELAY = 0.5  # Reduced since JSON API is faster

# ---------------------------------------------------------------------------
# Image Processing
# ---------------------------------------------------------------------------
IMAGE_SIZE = (512, 512)
MAX_IMAGE_SIZE = 1024

# ---------------------------------------------------------------------------
# Fashion Analysis
# ---------------------------------------------------------------------------
MAX_CLOTHING_ITEMS = 10
CONFIDENCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Matching Configuration
# ---------------------------------------------------------------------------
VISUAL_SIMILARITY_WEIGHT = 0.6
TEXT_SIMILARITY_WEIGHT = 0.4
MIN_SIMILARITY_SCORE = 0.25  # Slightly lower to show more real products

# ---------------------------------------------------------------------------
# UI Configuration
# ---------------------------------------------------------------------------
GRADIO_THEME = "soft"
MAX_RESULTS_DISPLAY = 12

# ---------------------------------------------------------------------------
# Device Configuration
# ---------------------------------------------------------------------------
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

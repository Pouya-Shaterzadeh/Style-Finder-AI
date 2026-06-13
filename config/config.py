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
# VLM — Llama 4 Scout 17B on GroqCloud — vision-capable model
# Free tier: no credit card required
# Get a free key at: https://console.groq.com
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

if GROQ_API_KEY:
    print(f"✓ Groq API key loaded (starts with: {GROQ_API_KEY[:8]}...)")
else:
    print("⚠️  GROQ_API_KEY not set — fashion analysis will not work.")

# ---------------------------------------------------------------------------
# Visual Similarity — patrickjohncyh/fashion-clip
# Trained on 800K+ fashion image-text pairs (vs generic openai/clip-vit)
# Runs on CPU — compatible with HF Spaces free tier (16 GB RAM)
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "patrickjohncyh/fashion-clip"

# HF token — used only to download fashion-CLIP weights (public model, optional)
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")

# ---------------------------------------------------------------------------
# Trendyol — internal JSON API (no HTML scraping, no bot detection)
# ---------------------------------------------------------------------------
TRENDYOL_BASE_URL    = "https://www.trendyol.com"
TRENDYOL_SEARCH_URL  = "https://www.trendyol.com/sr"
TRENDYOL_JSON_API_URL = (
    "https://public.trendyol.com/discovery-web-searchgw-service/api/"
    "infinite-scroll/product-search-with-typed-items"
)
MAX_SEARCH_RESULTS = 20
REQUEST_DELAY      = 0.5

# ---------------------------------------------------------------------------
# Image Processing
# ---------------------------------------------------------------------------
IMAGE_SIZE     = (512, 512)
MAX_IMAGE_SIZE = 1024

# ---------------------------------------------------------------------------
# Fashion Analysis
# ---------------------------------------------------------------------------
MAX_CLOTHING_ITEMS   = 10
CONFIDENCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
VISUAL_SIMILARITY_WEIGHT = 0.6
TEXT_SIMILARITY_WEIGHT   = 0.4
MIN_SIMILARITY_SCORE     = 0.25

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
GRADIO_THEME       = "soft"
MAX_RESULTS_DISPLAY = 12

# ---------------------------------------------------------------------------
# Device — auto-detect; HF Spaces free tier has no GPU → always "cpu"
# ---------------------------------------------------------------------------
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except ImportError:
    DEVICE = "cpu"

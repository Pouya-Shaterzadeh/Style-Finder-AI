"""
Image Processing Utilities
Handles image preprocessing and fashion-specific CLIP embeddings for visual similarity.

Uses patrickjohncyh/fashion-clip instead of the generic openai/clip-vit-base-patch32:
- Trained on 800K+ fashion image-text pairs
- 5-15% better recall on fashion product matching
- Same CLIPModel/CLIPProcessor API — drop-in replacement
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
import requests
from io import BytesIO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IMAGE_SIZE, MAX_IMAGE_SIZE, DEVICE, CLIP_MODEL_NAME

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: transformers not available. Visual similarity will be disabled.")


class ImageProcessor:
    """Handles image preprocessing and fashion-CLIP embedding generation."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self._load_clip()

    def _load_clip(self):
        """Load fashion-CLIP model for visual + text-image similarity."""
        if not CLIP_AVAILABLE:
            return
        try:
            print(f"Loading fashion-CLIP: {CLIP_MODEL_NAME}...")
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self.clip_model.eval()
            print(f"✅ fashion-CLIP loaded on {self.device.upper()}")
        except Exception as e:
            print(f"Error loading fashion-CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Resize and convert to RGB."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        width, height = image.size
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        return image

    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """Download and preprocess an image from a URL."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return self.preprocess_image(image)
        except Exception as e:
            print(f"Error loading image from URL {url}: {e}")
            return None

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def get_image_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Get normalized fashion-CLIP image embedding.

        Returns:
            1-D float32 array of shape (512,) or None on failure.
        """
        if self.clip_model is None:
            return None
        try:
            image = self.preprocess_image(image)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None

    # Keep old name as alias (used in trendyol_scraper.py)
    def get_clip_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        return self.get_image_embedding(image)

    def get_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get normalized fashion-CLIP text embedding.

        Args:
            text: English or Turkish fashion search query

        Returns:
            1-D float32 array of shape (512,) or None on failure.
        """
        if self.clip_model is None:
            return None
        try:
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(self.device)
            with torch.no_grad():
                feats = self.clip_model.get_text_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().numpy().flatten()
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return None

    # ------------------------------------------------------------------
    # Similarity scoring
    # ------------------------------------------------------------------

    def get_text_image_similarity(self, image: Image.Image, text: str) -> float:
        """
        Compute cosine similarity between an image and a text query.

        This is the primary scoring method used during product ranking:
        instead of comparing two product images (which requires downloading
        each product thumbnail), we compare the user's uploaded image
        against the search query text — instantaneous and accurate.

        Args:
            image: User-uploaded fashion image
            text:  Search query (e.g. "Kadın Lacivert Jean")

        Returns:
            Score in [0, 1] — higher = better match
        """
        img_emb = self.get_image_embedding(image)
        txt_emb = self.get_text_embedding(text)
        if img_emb is None or txt_emb is None:
            return 0.5  # Neutral score when embeddings unavailable
        return self._cosine_to_score(img_emb, txt_emb)

    def cosine_similarity(self,
                          embedding1: np.ndarray,
                          embedding2: np.ndarray) -> float:
        """Cosine similarity between two normalized embeddings, mapped to [0, 1]."""
        return self._cosine_to_score(embedding1, embedding2)

    def compare_images(self, image1: Image.Image, image2: Image.Image) -> float:
        """Compare two images by their CLIP embeddings."""
        e1 = self.get_image_embedding(image1)
        e2 = self.get_image_embedding(image2)
        if e1 is None or e2 is None:
            return 0.0
        return self._cosine_to_score(e1, e2)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_to_score(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity mapped from [-1, 1] → [0, 1]."""
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            cos_sim = np.dot(a, b) / (norm_a * norm_b)
            return float(max(0.0, min(1.0, (cos_sim + 1.0) / 2.0)))
        except Exception:
            return 0.0

"""
Image Processing Utilities
Handles image preprocessing and CLIP embeddings for visual similarity
"""

import torch
from PIL import Image
import numpy as np
from typing import Tuple, Optional, List
import requests
from io import BytesIO
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import IMAGE_SIZE, MAX_IMAGE_SIZE, DEVICE

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available. Visual similarity will be limited.")


class ImageProcessor:
    """Handles image preprocessing and embedding generation"""
    
    def __init__(self, device: str = DEVICE):
        """
        Initialize image processor
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self.clip_model = None
        self.clip_processor = None
        self._load_clip()
    
    def _load_clip(self):
        """Load CLIP model for visual similarity"""
        if not CLIP_AVAILABLE:
            return
        
        try:
            from config.config import CLIP_MODEL_NAME
            print(f"Loading CLIP model: {CLIP_MODEL_NAME}...")
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            self.clip_model.eval()
            print("CLIP model loaded successfully!")
        except Exception as e:
            print(f"Error loading CLIP model: {e}")
            self.clip_model = None
            self.clip_processor = None
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize if too large
        width, height = image.size
        if width > MAX_IMAGE_SIZE or height > MAX_IMAGE_SIZE:
            image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        return image
    
    def load_image_from_url(self, url: str) -> Optional[Image.Image]:
        """
        Load image from URL
        
        Args:
            url: Image URL
            
        Returns:
            PIL Image or None if failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            return self.preprocess_image(image)
        except Exception as e:
            print(f"Error loading image from URL {url}: {e}")
            return None
    
    def get_clip_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Get CLIP embedding for an image
        
        Args:
            image: PIL Image object
            
        Returns:
            CLIP embedding vector or None
        """
        if self.clip_model is None or self.clip_processor is None:
            return None
        
        try:
            image = self.preprocess_image(image)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy().flatten()
            
            return embedding
        except Exception as e:
            print(f"Error generating CLIP embedding: {e}")
            return None
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        try:
            # Normalize vectors
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is between 0 and 1
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception as e:
            print(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def compare_images(self, image1: Image.Image, image2: Image.Image) -> float:
        """
        Compare two images and return similarity score
        
        Args:
            image1: First PIL Image
            image2: Second PIL Image
            
        Returns:
            Similarity score (0-1)
        """
        embedding1 = self.get_clip_embedding(image1)
        embedding2 = self.get_clip_embedding(image2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        return self.cosine_similarity(embedding1, embedding2)


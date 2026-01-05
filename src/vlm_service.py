"""
Vision Language Model Service for Fashion Analysis

Supports both:
1. Local model loading (RECOMMENDED - works on HF Spaces and local dev)
2. Hugging Face Inference API (when models are available - currently most are not)

On Hugging Face Spaces:
- Model downloads ONCE when the Space starts
- Then stays loaded for all users
- Users just use the web app - no installation needed!
"""

import requests
from PIL import Image
import base64
from io import BytesIO
import json
from typing import Dict, List, Optional
import sys
import os
import time
import logging

# Setup logging for VLM service
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create logs directory if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

# File handler for VLM logs
vlm_log_file = os.path.join(log_dir, 'vlm_service.log')
file_handler = logging.FileHandler(vlm_log_file)
file_handler.setLevel(logging.DEBUG)

# Console handler for terminal output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Format for log messages
formatter = logging.Formatter('%(asctime)s - %(levelname)s - 🤖 [VLM] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import LLAVA_MODEL_NAME, MAX_CLOTHING_ITEMS, HF_API_TOKEN, USE_INFERENCE_API

# Try to import transformers for local model loading
LOCAL_MODEL_AVAILABLE = False
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    LOCAL_MODEL_AVAILABLE = True
    print("✓ Transformers available for local model loading")
except ImportError:
    print("⚠ Transformers not available - will try Inference API")

# Try to use InferenceClient (handles new router API automatically)
INFERENCE_CLIENT_AVAILABLE = False
try:
    from huggingface_hub import InferenceClient
    INFERENCE_CLIENT_AVAILABLE = True
except ImportError:
    pass


class VLMService:
    """Service for analyzing fashion images using Vision Language Models
    
    Supports:
    - Local model loading (RECOMMENDED - works on HF Spaces, model downloads once)
    - Inference API (fallback - most models unavailable on free tier)
    """
    
    def __init__(self, model_name: str = LLAVA_MODEL_NAME, use_api: bool = USE_INFERENCE_API):
        """
        Initialize the VLM service
        
        Args:
            model_name: Hugging Face model identifier
            use_api: If True, use Inference API. If False, load model locally (recommended).
        """
        self.model_name = model_name
        self.use_api = use_api
        self.local_model = None
        self.local_processor = None
        self.use_client = False
        self.model_type = None
        
        # Determine device
        if LOCAL_MODEL_AVAILABLE:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        
        # Try local model first (recommended for reliability)
        if not use_api and LOCAL_MODEL_AVAILABLE:
            self._load_local_model()
        elif use_api:
            self._setup_api()
        else:
            logger.warning("Local models require transformers. Install with: pip install transformers torch")
            logger.warning("Falling back to API...")
            self.use_api = True
            self._setup_api()
    
    def _load_local_model(self):
        """Load model locally - downloads once, then cached forever"""
        logger.info(f"Loading model: {self.model_name}")
        logger.info("(First run downloads model, then it's cached for instant loading)")
        
        try:
            import torch
            
            # Load BLIP model
            if "blip" in self.model_name.lower():
                logger.info("Loading BLIP image captioning model...")
                self.local_processor = BlipProcessor.from_pretrained(self.model_name)
                self.local_model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.model_type = "blip"
                logger.info(f"✅ BLIP model loaded successfully on {self.device.upper()}")
            else:
                # Try generic Vision2Seq for other models
                logger.info(f"Loading {self.model_name}...")
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.local_processor = AutoProcessor.from_pretrained(self.model_name)
                self.local_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.model_type = "generic"
                logger.info(f"✅ Model loaded successfully on {self.device.upper()}")
                
        except Exception as e:
            logger.error(f"Could not load local model: {e}")
            logger.warning("Falling back to API (may not work)...")
            self.use_api = True
            self._setup_api()
    
    def _setup_api(self):
        """Setup API endpoints (fallback - most models unavailable)"""
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.api_url_fallback = f"https://router.huggingface.co/inference/{self.model_name}"
        self.headers = {"Content-Type": "application/json"}
        if HF_API_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
        
        if INFERENCE_CLIENT_AVAILABLE and HF_API_TOKEN:
            try:
                self.client = InferenceClient(token=HF_API_TOKEN)
                self.use_client = True
                print(f"Initialized with InferenceClient: {self.model_name}")
            except Exception as e:
                print(f"InferenceClient failed: {e}")
        
        print(f"⚠ Using API mode - most vision models are NOT available on free tier")
    
    def _create_fashion_prompt(self) -> str:
        """
        Create a detailed prompt for fashion analysis
        
        Returns:
            Formatted prompt string
        """
        prompt = """Analyze this fashion image in detail. Extract all clothing items, accessories, and fashion elements. For each item, identify:

1. Item type (e.g., dress, shirt, pants, jacket, shoes, bag, jewelry)
2. Color(s) - be specific (e.g., navy blue, burgundy, cream white)
3. Pattern/Design (e.g., solid, striped, floral, polka dot, plaid)
4. Style (e.g., casual, formal, sporty, bohemian, minimalist)
5. Material/Fabric hints (e.g., denim, cotton, silk, leather)
6. Key features (e.g., long sleeves, V-neck, high-waisted, ankle-length)

Format your response as a JSON object with this structure:
{
  "items": [
    {
      "type": "item_type",
      "color": "primary_color",
      "pattern": "pattern_type",
      "style": "style_description",
      "material": "material_hint",
      "features": ["feature1", "feature2"],
      "description": "detailed_description"
    }
  ],
  "overall_style": "overall_outfit_style",
  "occasion": "suitable_occasion"
}

Be thorough and identify ALL visible clothing items and accessories. Return ONLY valid JSON."""
        
        return prompt
    
    def image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL Image to base64 string
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded string
        """
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    
    def _analyze_with_local_model(self, image: Image.Image) -> str:
        """
        Analyze image using locally loaded model
        
        Args:
            image: PIL Image object
            
        Returns:
            Generated caption/description
        """
        try:
            import torch
            logger.info("Analyzing image with local model...")
            
            if self.model_type == "blip":
                # BLIP conditional image captioning - use text prompts to guide toward fashion
                # Generate multiple captions with different prompts for better coverage
                captions = []
                
                # IMPROVED: Use smarter, more targeted prompts
                # Key principles:
                # 1. Start with neutral observation prompts
                # 2. Ask about what's ACTUALLY visible, not assumptions
                # 3. Include prompts for layered clothing (shirt under sweater)
                # 4. Include prompts for visible accessories (watch on wrist)
                # 5. Ask about visibility before assuming items exist
                
                fashion_prompts = [
                    # General detection - let the model describe freely
                    "a photo of a person wearing",                       # General clothing detection
                    "this is a photo of a woman wearing",                # Female detection + clothing  
                    "this is a photo of a man wearing",                  # Male detection + clothing
                    
                    # Specific clothing items with color focus
                    "the color of the top is",                           # Top/sweater color
                    "the bottom clothing in this photo is",              # Pants/jeans/skirt detection
                    "the pants color is",                                # Specific pants color (helps detect cream/beige)
                    "the person is wearing jeans that are",              # Jeans detection (blue/denim)
                    
                    # Footwear detection
                    "the shoes in this photo are",                       # Direct shoe detection
                    
                    # Layered clothing detection (shirt under sweater/jacket)
                    "under the sweater there is a",                      # Detect shirt collar under sweater
                    "the collar visible is",                             # Detect visible collar
                    
                    # Accessories - focused detection
                    "the watch on the wrist is",                         # Watch detection (direct mention of watch)
                    "the person is wearing a watch that is",             # Alternative watch prompt
                    
                    # Ask about body parts visible to avoid hallucinating hidden items
                    "the person's feet in this photo show",              # Check if feet/shoes visible (more specific)
                    
                    # Style
                    "the outfit style is",                               # Overall style
                ]
                
                for prompt in fashion_prompts:
                    inputs = self.local_processor(image, text=prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        output = self.local_model.generate(
                            **inputs,
                            max_new_tokens=80,
                            num_beams=5,
                            early_stopping=True,
                            repetition_penalty=1.2
                        )
                    
                    caption = self.local_processor.decode(output[0], skip_special_tokens=True)
                    captions.append(caption)
                    logger.debug(f"Prompt '{prompt}': {caption}")
                
                # Combine all captions for comprehensive extraction
                combined_caption = " | ".join(captions)
                logger.info(f"✅ Combined fashion caption: {combined_caption[:300]}...")
                return combined_caption
            else:
                # Generic model
                inputs = self.local_processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    output = self.local_model.generate(**inputs, max_new_tokens=200)
                
                text = self.local_processor.decode(output[0], skip_special_tokens=True)
                logger.info(f"✅ Generated text: {text[:100]}...")
                return text
                
        except Exception as e:
            logger.error(f"Local model analysis failed: {e}")
            return None
    
    def analyze_fashion_image(self, image: Image.Image) -> Dict:
        """
        Analyze a fashion image using local model or API
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary containing extracted fashion attributes
        """
        try:
            generated_text = None
            is_llava_model = "llava" in self.model_name.lower()
            
            # Use LOCAL MODEL if available (recommended)
            if self.local_model is not None:
                generated_text = self._analyze_with_local_model(image)
            
            # Fall back to API if local model not available
            if generated_text is None and self.use_api:
                # Try InferenceClient's image_to_text first
                if self.use_client and not is_llava_model:
                    try:
                        logger.info("Trying InferenceClient image_to_text method...")
                        result = self.client.image_to_text(image=image, model=self.model_name)
                        if result:
                            generated_text = result
                            logger.info("✅ Successfully used InferenceClient image_to_text")
                    except Exception as e:
                        logger.warning(f"InferenceClient image_to_text failed: {e}")
                        logger.info("Falling back to direct API calls...")
                
                # For LLaVA or if InferenceClient failed, use direct API calls
                if not generated_text:
                    if is_llava_model:
                        prompt = self._create_fashion_prompt()
                        logger.info("Sending request to Hugging Face Inference API with custom prompt...")
                    else:
                        prompt = None
                        logger.info("Sending request to Hugging Face Inference API for image captioning...")
                    
                    try:
                        generated_text = self._call_api_direct(image, prompt)
                    except requests.exceptions.Timeout:
                        logger.error("Request timeout - the model might be taking longer than expected")
                        return {
                            "items": [],
                            "overall_style": "unknown",
                            "occasion": "unknown",
                            "error": "Request timeout - please try again"
                        }
            
            # Extract fashion data from response
            # For BLIP models, we get a caption that we need to parse
            # For LLaVA models, we should get structured JSON
            if "llava" in self.model_name.lower():
                # LLaVA should return structured JSON
                fashion_data = self._parse_response(generated_text)
            else:
                # BLIP returns a caption - we need to extract fashion info from it
                fashion_data = self._extract_fashion_from_caption(generated_text)
            
            return fashion_data
            
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {
                "items": [],
                "overall_style": "unknown",
                "occasion": "unknown",
                "error": str(e)
            }
    
    def _call_api_direct(self, image: Image.Image, prompt: str = None) -> str:
        """
        Make direct API call to Hugging Face Inference API
        
        Args:
            image: PIL Image object
            prompt: Text prompt (optional, for models that support it like LLaVA)
            
        Returns:
            Generated text response
        """
        # Convert image to base64
        image_base64 = self.image_to_base64(image)
        
        # Check if model supports custom prompts (like LLaVA) or just image captioning (like BLIP)
        is_llava_model = "llava" in self.model_name.lower()
        
        if is_llava_model and prompt:
            # For LLaVA models, use image + text prompt format
            payload = {
                "inputs": {
                    "image": image_base64,
                    "text": prompt
                },
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "do_sample": True
                }
            }
        else:
            # For BLIP and other image-to-text models, just send the image
            # They generate captions automatically
            payload = {
                "inputs": image_base64
            }
        
        # Make API request to router endpoint
        print(f"Making request to: {self.api_url}")
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        # If old endpoint returns 410 (deprecated), check if response still has data
        # Sometimes "deprecated" endpoints still work, they just return 410 with a warning
        if response.status_code == 410:
            # Check if the response actually contains data despite 410 status
            try:
                result = response.json()
                if result and not result.get("error"):
                    # 410 but got valid data - endpoint still works!
                    print("Note: Endpoint returned 410 (deprecated) but request succeeded")
                    # Process the response normally
                    if isinstance(result, dict):
                        generated_text = result.get("generated_text", "") or result.get("text", "")
                    elif isinstance(result, str):
                        generated_text = result
                    else:
                        generated_text = str(result)
                    return generated_text
            except:
                pass
            
            # If 410 with no data, try router endpoint
            print("Old endpoint returned 410 with no data, trying router endpoint...")
            print(f"Fallback URL: {self.api_url_fallback}")
            response = requests.post(
                self.api_url_fallback,
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code not in [200, 503]:
                print(f"Router endpoint returned {response.status_code}")
                print("Both endpoints failed. The model might not be available on Inference API.")
                print("Note: LLaVA models may not be available on the free Inference API.")
                print("Consider:")
                print("  1. Using a different vision-language model (e.g., BLIP, CLIP-based)")
                print("  2. Running the model locally (requires ~7GB download)")
                print("  3. Using a paid Hugging Face Inference API plan")
        
        # Handle response
        if response.status_code == 200:
            result = response.json()
            
            # Extract generated text from response
            if isinstance(result, dict):
                generated_text = result.get("generated_text", "")
                if not generated_text:
                    generated_text = result.get("text", "")
                    if not generated_text and isinstance(result.get("output"), str):
                        generated_text = result["output"]
            elif isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, str):
                generated_text = result
            else:
                generated_text = str(result)
            
            return generated_text
            
        elif response.status_code == 503:
            # Model is loading, wait and retry
            print("Model is loading, waiting 30 seconds and retrying...")
            time.sleep(30)
            return self._call_api_direct(image, prompt)  # Retry once
            
        else:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(error_msg)
            raise Exception(error_msg)
    
    def _parse_response(self, response: str) -> Dict:
        """
        Parse the model response and extract JSON
        
        Args:
            response: Raw model response text
            
        Returns:
            Parsed fashion data dictionary
        """
        try:
            # Try to find JSON in the response
            # Look for JSON object boundaries
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                fashion_data = json.loads(json_str)
                
                # Validate and clean data
                if "items" not in fashion_data:
                    fashion_data["items"] = []
                
                # Ensure gender field exists
                if "gender" not in fashion_data:
                    fashion_data["gender"] = "unknown"
                
                # Limit number of items
                if len(fashion_data.get("items", [])) > MAX_CLOTHING_ITEMS:
                    fashion_data["items"] = fashion_data["items"][:MAX_CLOTHING_ITEMS]
                
                return fashion_data
            else:
                # Fallback: try to extract information manually
                return self._extract_fallback_info(response)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, try fallback extraction
            return self._extract_fallback_info(response)
    
    def _extract_fashion_from_caption(self, caption: str) -> Dict:
        """
        Extract fashion information from a BLIP-generated caption.
        Enhanced to better detect clothing items and colors.
        
        Args:
            caption: Image caption text from BLIP
            
        Returns:
            Dictionary with extracted fashion data
        """
        import re
        
        # Initialize result structure
        result = {
            "items": [],
            "overall_style": "casual",
            "occasion": "everyday",
            "gender": "unknown"  # Will be detected from caption
        }
        
        caption_lower = caption.lower()
        found_items = set()  # Track what we've found to avoid duplicates
        
        # IMPORTANT: Strip BLIP prompt prefixes from segments to avoid false pattern matches
        # BLIP echoes our prompts back, e.g., "under the sweater there is a black jacket"
        # The word "sweater" in the prompt causes false sweater detection!
        # We need to analyze only the RESPONSE part, not the prompt
        prompt_prefixes_to_strip = [
            "under the sweater there is a",
            "the collar visible is",
            "the watch on the wrist is",
            "the person is wearing a watch that is",
            "the shoes in this photo are",
            "the person's feet in this photo show",
            "a photo of a person wearing",
            "this is a photo of a woman wearing",
            "this is a photo of a man wearing",
            "the color of the top is",
            "the bottom clothing in this photo is",
            "the pants color is",
            "the person is wearing jeans that are",
            "the outfit style is",
        ]
        
        # Create a cleaned caption where prompt prefixes are stripped from each segment
        cleaned_segments = []
        for segment in caption_lower.split(" | "):
            cleaned_segment = segment
            for prefix in prompt_prefixes_to_strip:
                if segment.startswith(prefix):
                    # Remove the prompt prefix, keep only BLIP's response
                    cleaned_segment = segment[len(prefix):].strip()
                    break
            cleaned_segments.append(cleaned_segment)
        
        # Use cleaned caption for item pattern matching (no prompt echoes)
        caption_cleaned = " | ".join(cleaned_segments)
        logger.debug(f"Cleaned caption (prompts stripped): {caption_cleaned[:200]}...")
        
        # Detect gender from caption - IMPROVED LOGIC
        # IMPORTANT: We use BOTH male and female prompts, so we need to look at:
        # 1. The FIRST general caption ("a photo of a person wearing") - neutral
        # 2. Visual cues like long hair, body shape, clothing style
        # 3. Gender-specific clothing items
        
        # Split the caption to analyze each part separately
        caption_parts = caption_lower.split(" | ")
        
        # Look for visual gender indicators in the FIRST (neutral) caption
        first_caption = caption_parts[0] if caption_parts else caption_lower
        
        # Check for visual indicators (these appear in neutral captions)
        female_visual_indicators = [
            r"\blong\s+hair\b", r"\bponytail\b", r"\bbraid\b", r"\bcurls\b",
            r"\bdress\b", r"\bskirt\b", r"\bheels\b", r"\bblouse\b",
            r"\bpurse\b", r"\bhandbag\b", r"\bearrings\b", r"\bnecklace\b",
            r"\bfeminine\b", r"\bwoman\b", r"\bgirl\b", r"\blady\b",
        ]
        male_visual_indicators = [
            r"\bbeard\b", r"\bmustache\b", r"\bshort\s+hair\b", 
            r"\btie\b", r"\bsuit\s+and\s+tie\b", r"\bmasculine\b",
            r"\bman\b", r"\bguy\b", r"\bgentleman\b", r"\bboy\b",
        ]
        
        female_visual_count = sum(1 for pattern in female_visual_indicators if re.search(pattern, caption_lower))
        male_visual_count = sum(1 for pattern in male_visual_indicators if re.search(pattern, caption_lower))
        
        # Count explicit gender mentions ONLY in non-prompted captions
        # Skip the "this is a photo of a woman/man wearing" responses since those are echoes
        male_count = 0
        female_count = 0
        for part in caption_parts:
            # Skip parts that start with "this is a photo of a" (our prompted responses)
            if part.strip().startswith("this is a photo of a"):
                continue
            male_count += len(re.findall(r"\b(man|men|male|guy|boy|gentleman|his|he)\b", part))
            female_count += len(re.findall(r"\b(woman|women|female|girl|lady|her|she)\b", part))
        
        logger.debug(f"Gender detection: male_count={male_count}, female_count={female_count}, female_visual={female_visual_count}, male_visual={male_visual_count}")
        
        # Also check for gender-specific clothing items (stronger indicators)
        female_clothing = ["dress", "skirt", "heels", "blouse", "purse", "handbag", "lipstick"]
        male_clothing = ["suit", "tie", "beard", "mustache"]
        
        female_clothing_count = sum(1 for word in female_clothing if word in caption_lower)
        male_clothing_count = sum(1 for word in male_clothing if word in caption_lower)
        
        # Weight: visual indicators (3x), clothing items (2x), mentions (1x)
        total_female = female_count + (female_clothing_count * 2) + (female_visual_count * 3)
        total_male = male_count + (male_clothing_count * 2) + (male_visual_count * 3)
        
        logger.debug(f"Weighted gender: total_female={total_female}, total_male={total_male}")
        
        if total_female > total_male:
            result["gender"] = "female"
        elif total_male > total_female:
            result["gender"] = "male"
        else:
            # If equal or both zero, keep as unknown
            # Don't assume gender - it's better to have no gender prefix than wrong one
            result["gender"] = "unknown"
            logger.debug(f"Gender kept as unknown (no clear indicators)")
        
        logger.info(f"Detected gender: {result['gender']}")
        
        # Comprehensive clothing detection
        # IMPORTANT: Order matters! More specific patterns should come first
        clothing_patterns = {
            # Tops - Shirt (collared) should be detected separately from T-shirt (no collar)
            # STRICT patterns - only match if shirt is explicitly described as a visible garment
            # Avoid generic "shirt" which might appear in any caption
            "shirt": [r"\bdress\s*shirt\b", r"\bbutton[\s-]*down\s+shirt\b", r"\bbutton[\s-]*up\s+shirt\b", 
                      r"\bcollared\s+shirt\b", r"\bwhite\s+shirt\b", r"\bblue\s+shirt\b", 
                      r"\bwearing\s+a\s+shirt\b", r"\bshirt\s+under\b", r"\bshirt\s+underneath\b"],
            # T-shirt = no collar, crew neck, casual tee
            "t-shirt": [r"\bt-shirt\b", r"\btee\b", r"\btshirt\b", r"\bt\s*shirt\b", r"\bcrew-neck\b", r"\bcrew\s*neck\b"],
            "blouse": [r"\bblouse\b"],
            "sweater": [r"\bsweater\b", r"\bpullover\b", r"\bjumper\b", r"\bcardigan\b", r"\bknit\b"],
            # Hoodie - including zip-up hoodie
            "hoodie": [r"\bhoodie\b", r"\bhood\b", r"\bzip[\s-]*up\s*hoodie\b", r"\bzipped\s*hoodie\b"],
            # Jackets/Outerwear - including zip-up jacket
            "jacket": [r"\bjacket\b", r"\bblazer\b", r"\bbomber\b", r"\bzip[\s-]*up\s*jacket\b", r"\bzipped\s*jacket\b"],
            # Coat - including buttoned coat
            "coat": [r"\bcoat\b", r"\bovercoat\b", r"\bbuttoned\s*coat\b"],
            # Bottoms
            "pants": [r"\bpants\b", r"\btrousers\b", r"\bslacks\b"],
            "jeans": [r"\bjeans\b", r"\bdenim\b"],
            "shorts": [r"\bshorts\b"],
            "skirt": [r"\bskirt\b"],
            # Full body
            "dress": [r"\bdress\b", r"\bgown\b"],
            "suit": [r"\bsuit\b"],
            # Footwear
            "shoes": [r"\bshoes\b", r"\bsneakers\b", r"\bboots\b", r"\bheels\b", r"\bsandals\b", r"\bloafers\b",
                      r"\bflats\b", r"\bpumps\b", r"\bmoccasins\b", r"\bslippers\b", r"\bfootwear\b",
                      r"\btrainers\b", r"\bkicks\b", r"\bwedges\b", r"\bmules\b", r"\bplatforms\b"],
            # Accessories
            "bag": [r"\bbag\b", r"\bpurse\b", r"\bbackpack\b", r"\bhandbag\b"],
            "hat": [r"\bhat\b", r"\bcap\b", r"\bbeanie\b"],
            "watch": [r"\bwatch\b", r"\bwristwatch\b"],
            "glasses": [r"\bglasses\b", r"\bsunglasses\b"],
            "belt": [r"\bbelt\b", r"\bwaist\s*belt\b"],
        }
        
        # Color detection with more options (including metallic for accessories)
        colors = {
            "white": [r"\bwhite\b"],
            "cream": [r"\bcream\b", r"\bivory\b", r"\boff-white\b", r"\beggshell\b"],
            "beige": [r"\bbeige\b", r"\btan\b", r"\bsand\b", r"\bcamel\b"],
            "khaki": [r"\bkhaki\b"],
            "taupe": [r"\btaupe\b", r"\bgreige\b", r"\bmushroom\b"],
            "olive": [r"\bolive\b", r"\bmilitary\s*green\b", r"\barmy\s*green\b"],
            "black": [r"\bblack\b"],
            # Multi-color patterns (check these FIRST - they're more specific)
            "black and white": [r"\bblack\s+and\s+white\b", r"\bblack\s*&\s*white\b", r"\bb&w\b"],
            # Blue - including light blue, denim blue
            "light blue": [r"\blight\s*blue\b", r"\bsky\s*blue\b", r"\bbaby\s*blue\b", r"\bpowder\s*blue\b"],
            "blue": [r"\bblue\b", r"\bnavy\b", r"\bdenim\b", r"\bjean\b"],
            "red": [r"\bred\b", r"\bburgundy\b", r"\bmaroon\b"],
            "green": [r"\bgreen\b", r"\bforest\b", r"\bmint\b", r"\bemerald\b"],
            "brown": [r"\bbrown\b", r"\bchocolate\b", r"\bcoffee\b"],
            "gray": [r"\bgray\b", r"\bgrey\b", r"\bcharcoal\b"],
            "pink": [r"\bpink\b", r"\bblush\b", r"\brose\b"],
            "yellow": [r"\byellow\b", r"\bmustard\b"],
            "purple": [r"\bpurple\b", r"\blavender\b", r"\bviolet\b"],
            "orange": [r"\borange\b", r"\bcoral\b", r"\bterracotta\b"],
            # Metallic colors (common for watches, jewelry, accessories)
            "silver": [r"\bsilver\b", r"\bmetallic\b", r"\bsteel\b", r"\bchrome\b"],
            "gold": [r"\bgold\b", r"\bgolden\b"],
            "rose gold": [r"\brose\s*gold\b"],
        }
        
        # Pattern detection (striped, plaid, etc.)
        pattern_types = {
            "striped": [r"\bstripe[sd]?\b", r"\bstriped\b", r"\bstripy\b", r"\bpinstripe[sd]?\b"],
            "plaid": [r"\bplaid\b", r"\btartan\b", r"\bcheckered\b", r"\bchecks?\b"],
            "floral": [r"\bfloral\b", r"\bflower[sy]?\b"],
            "polka dot": [r"\bpolka\s*dot[s]?\b", r"\bdotted\b", r"\bspots?\b"],
            "solid": [r"\bsolid\b", r"\bplain\b"],
        }
        
        # Helper function to detect pattern for an item
        def find_pattern_for_item(item_type: str, caption_lower: str) -> str:
            """Detect pattern (striped, plaid, etc.) for an item from caption
            
            IMPORTANT: Patterns only apply to clothing items, not accessories like watches, belts, glasses
            Jeans and pants are typically solid - patterns apply to tops mainly
            
            CRITICAL: Only detect pattern if it's mentioned WITH the specific item,
            not just anywhere in the caption (to avoid cross-contamination)
            """
            # Items that CANNOT have patterns (accessories) or are always solid
            no_pattern_items = ["watch", "belt", "glasses", "sunglasses", "bag", "hat", "shoes", "sneakers", "boots",
                               "jeans", "pants", "shorts", "skirt"]  # Bottom wear is typically solid
            if item_type in no_pattern_items:
                return "solid"
            
            # Split caption into segments and find item-specific segments
            caption_segments = caption_lower.split(" | ")
            item_segments = [seg for seg in caption_segments if item_type in seg]
            
            # If no specific segment, use full caption but with strict locality
            search_text = " ".join(item_segments) if item_segments else caption_lower
            
            # Check for pattern mentions SPECIFICALLY with this item
            for pattern_name, pattern_regexes in pattern_types.items():
                for regex in pattern_regexes:
                    # Check if pattern is mentioned in item-specific context
                    if re.search(regex, search_text):
                        # For striped, require it to be explicitly with the item
                        if pattern_name == "striped":
                            # Check if "striped [item]" or "[item] striped" or "[item] with stripes"
                            striped_with_item = (
                                re.search(rf"\bstriped?\s+{item_type}", search_text) or
                                re.search(rf"{item_type}\s+(?:is\s+|with\s+)?striped?", search_text) or
                                re.search(rf"{item_type}.*stripes", search_text)
                            )
                            # Also check for "black and white [item]" pattern
                            bw_with_item = re.search(rf"black\s+and\s+white\s+(?:striped?\s+)?{item_type}", search_text)
                            
                            if striped_with_item or bw_with_item:
                                return "striped"
                        else:
                            # For other patterns, also require proximity to item
                            if re.search(rf"{regex}.*{item_type}|{item_type}.*{regex}", search_text):
                                return pattern_name
            
            return "solid"
        
        # Texture/Material detection
        textures = {
            "knit": [r"\bknit\b", r"\bknitted\b", r"\bknitwear\b"],
            "cotton": [r"\bcotton\b"],
            "denim": [r"\bdenim\b", r"\bjean\b"],
            "leather": [r"\bleather\b"],
            "silk": [r"\bsilk\b", r"\bsilky\b", r"\bsatin\b"],
            "wool": [r"\bwool\b", r"\bwoolen\b", r"\bcashmere\b"],
            "linen": [r"\blinen\b"],
            "velvet": [r"\bvelvet\b"],
            "suede": [r"\bsuede\b"],
            "fleece": [r"\bfleece\b"],
            "chiffon": [r"\bchiffon\b"],
            "tweed": [r"\btweed\b"],
            "corduroy": [r"\bcorduroy\b"],
        }
        
        # Closure type detection (buttons, zippers)
        closures = {
            "buttoned": [r"\bbutton\b", r"\bbuttoned\b", r"\bbuttons\b", r"\bbutton[\s-]*up\b", r"\bbutton[\s-]*down\b"],
            "zippered": [r"\bzip\b", r"\bzipper\b", r"\bzipped\b", r"\bzip[\s-]*up\b", r"\bzippered\b"],
        }
        
        # Helper function to find closure type for an item
        def find_closure_for_item(item_type: str, caption_lower: str) -> str:
            """Find closure type (button/zipper) ONLY for items that can have closures
            
            IMPORTANT: Only sweaters, jackets, hoodies, coats, and shirts can have closures.
            Watches, pants, belts, etc. NEVER have closures.
            """
            
            # Items that CAN have closures
            closure_items = ["sweater", "jacket", "hoodie", "coat", "shirt"]
            
            # Items that CANNOT have closures - return immediately
            if item_type not in closure_items:
                return "unknown"
            
            # Now check for closure patterns, but only near the item type
            # Check for ZIPPER first (more visible/distinct)
            zip_patterns = [
                rf"\b{item_type}\s+with\s+zip",
                rf"\bzip[\s-]*up\s+{item_type}",
                rf"\bzippered\s+{item_type}",
                rf"\b{item_type}\s+has\s+a\s+zipper",
            ]
            for pattern in zip_patterns:
                if re.search(pattern, caption_lower):
                    print(f"    ZIPPER detected for {item_type}")
                    return "zippered"
            
            # For sweaters, check for crew-neck/pullover (no closure) FIRST
            if item_type == "sweater":
                # Crew-neck and pullover sweaters have NO closure
                # Also check for patterns that indicate no closure
                no_closure_patterns = [
                    r"\bcrew[\s-]*neck",
                    r"\bpullover\b",
                    r"\bcrew\s+neck\b",
                    r"\bround\s+neck\b",  # Round neck = no closure
                    r"\bsweater\s+with\s+no\b",  # "sweater with no zip/buttons"
                ]
                if any(re.search(p, caption_lower) for p in no_closure_patterns):
                    print(f"    Crew-neck/pullover/round-neck sweater -> NO closure")
                    return "unknown"
                
                # Check for explicit zip-up patterns (must be very specific)
                explicit_zip_patterns = [
                    rf"\bzip[\s-]*up\s+{item_type}",
                    rf"\bzippered\s+{item_type}",
                    rf"\b{item_type}\s+with\s+zip",
                    rf"\b{item_type}\s+has\s+a\s+zipper",
                ]
                for pattern in explicit_zip_patterns:
                    if re.search(pattern, caption_lower):
                        print(f"    Explicit ZIP-UP sweater detected")
                        return "zippered"
                
                # Only check for zip if NOT crew-neck/pullover AND zip is near sweater
                if re.search(r"\bzip|\bzipper", caption_lower) and re.search(rf"\b{item_type}\b", caption_lower):
                    # Check if they're close together (within 30 chars)
                    zip_matches = list(re.finditer(r"\bzip|\bzipper", caption_lower))
                    item_matches = list(re.finditer(rf"\b{item_type}\b", caption_lower))
                    for zip_match in zip_matches:
                        for item_match in item_matches:
                            if abs(zip_match.start() - item_match.start()) < 30:
                                print(f"    ZIPPER near {item_type} -> ZIPPERED")
                                return "zippered"
                
                # If no explicit closure mentioned, default to "unknown" (crew-neck/pullover)
                # Don't assume it has a closure
                print(f"    No explicit closure for sweater -> NO closure (likely crew-neck/pullover)")
                return "unknown"
            
            # For jackets/hoodies, check if zip is mentioned near them
            if item_type in ["jacket", "hoodie"]:
                if re.search(r"\bzip|\bzipper", caption_lower) and re.search(rf"\b{item_type}\b", caption_lower):
                    # Check if they're close together (within 30 chars)
                    zip_matches = list(re.finditer(r"\bzip|\bzipper", caption_lower))
                    item_matches = list(re.finditer(rf"\b{item_type}\b", caption_lower))
                    for zip_match in zip_matches:
                        for item_match in item_matches:
                            if abs(zip_match.start() - item_match.start()) < 30:
                                print(f"    ZIPPER near {item_type} -> ZIPPERED")
                                return "zippered"
            
            # Check for BUTTONS
            button_patterns = [
                rf"\b{item_type}\s+with\s+buttons",
                rf"\bbuttoned\s+{item_type}",
                rf"\bbutton[\s-]*up\s+{item_type}",
                rf"\b{item_type}\s+has\s+buttons",
            ]
            for pattern in button_patterns:
                if re.search(pattern, caption_lower):
                    print(f"    BUTTON detected for {item_type}")
                    return "buttoned"
            
            # Check for CARDIGAN (typically buttoned, but not always)
            if item_type == "sweater" and re.search(r"\bcardigan\b", caption_lower):
                # Only return buttoned if no zip was mentioned
                if not re.search(r"\bzip", caption_lower):
                    print(f"    Cardigan without zip -> BUTTONED")
                    return "buttoned"
            
            # DEFAULT: No closure
            return "unknown"
        
        # Helper function to extract color combinations from caption for an item
        def extract_color_description(item_type: str, item_patterns: list, caption_lower: str) -> str:
            """Extract the ACTUAL color description from the caption for an item.
            
            This captures color combinations like "black and white", "blue and gray", etc.
            instead of just single colors.
            
            IMPORTANT: Search within each caption segment (separated by |) to avoid
            picking up colors from unrelated items.
            
            Returns: Color description string (e.g., "black and white", "light blue", "red")
            """
            # Color words to look for
            color_words = [
                "white", "black", "blue", "red", "green", "yellow", "pink", "purple", 
                "orange", "brown", "gray", "grey", "beige", "cream", "navy", "silver", 
                "gold", "olive", "khaki", "tan", "maroon", "burgundy", "teal", "coral",
                "ivory", "off-white", "eggshell", "ecru",  # Cream-like colors
                "light", "dark", "bright", "pale", "deep"  # Color modifiers
            ]
            
            # Split caption into segments (each prompt response is separated by |)
            caption_segments = caption_lower.split(" | ")
            
            # PRIORITY: Check for specific item color prompts first (e.g., "the pants color is")
            # These are more accurate than general captions
            if item_type in ["pants", "jeans", "trousers"]:
                for segment in caption_segments:
                    # Look for "the pants color is X" prompt response
                    pants_color_match = re.search(r"the pants color is\s+(\w+)", segment)
                    if pants_color_match:
                        color = pants_color_match.group(1).lower()
                        if color in color_words:
                            logger.debug(f"Found pants color from direct prompt: '{color}'")
                            return color
            
            # Find segments that mention this item type
            item_segments = []
            for segment in caption_segments:
                for pattern in item_patterns:
                    if re.search(pattern, segment):
                        item_segments.append(segment)
                        break
            
            # If no segments mention this item, search the full caption but with strict locality
            if not item_segments:
                item_segments = [caption_lower]
            
            # Search for colors in item-specific segments
            for segment in item_segments:
                # Look for color + item patterns (most reliable)
                # e.g., "black and white striped sweater", "light blue jeans"
                for pattern in item_patterns:
                    item_match = re.search(pattern, segment)
                    if item_match:
                        # Get text BEFORE the item (where color usually is)
                        text_before = segment[:item_match.start()]
                        
                        # Look for color combination immediately before item
                        # e.g., "black and white striped sweater"
                        combo_match = re.search(
                            r"\b((?:light|dark|bright|pale|deep)\s+)?(" + "|".join(color_words[:20]) + r")\s+and\s+((?:light|dark|bright|pale|deep)\s+)?(" + "|".join(color_words[:20]) + r")\b",
                            text_before[-50:]  # Only look at last 50 chars before item
                        )
                        if combo_match:
                            color_desc = combo_match.group(0).strip()
                            logger.debug(f"Found color combo '{color_desc}' for {item_type} in segment")
                            return color_desc
                        
                        # Look for modified color (light blue, dark green)
                        modified_match = re.search(
                            r"\b(light|dark|bright|pale|deep)\s+(blue|green|gray|grey|brown|pink|purple|red|yellow)\b",
                            text_before[-40:]
                        )
                        if modified_match:
                            color_desc = modified_match.group(0).strip()
                            logger.debug(f"Found modified color '{color_desc}' for {item_type}")
                            return color_desc
                        
                        # Look for single color immediately before item
                        single_match = re.search(
                            r"\b(" + "|".join(color_words[:20]) + r")\s*$",
                            text_before[-30:]
                        )
                        if single_match:
                            color_desc = single_match.group(1).strip()
                            logger.debug(f"Found single color '{color_desc}' for {item_type}")
                            return color_desc
                
                # Also check for "item is/are [color]" pattern
                # e.g., "the jeans are light blue"
                for pattern in item_patterns:
                    is_pattern = re.search(
                        rf"{pattern}\s+(?:is|are|that\s+are)\s+((?:light|dark|bright|pale|deep)\s+)?(" + "|".join(color_words[:20]) + r")",
                        segment
                    )
                    if is_pattern:
                        modifier = is_pattern.group(1) if is_pattern.group(1) else ""
                        color = is_pattern.group(2)
                        color_desc = f"{modifier}{color}".strip()
                        logger.debug(f"Found color '{color_desc}' after {item_type} (is/are pattern)")
                        return color_desc
                    
                    # Check for "item that are [color] and [color]"
                    combo_after = re.search(
                        rf"{pattern}\s+(?:is|are|that\s+are)\s+(" + "|".join(color_words[:20]) + r")\s+and\s+(" + "|".join(color_words[:20]) + r")",
                        segment
                    )
                    if combo_after:
                        color_desc = f"{combo_after.group(1)} and {combo_after.group(2)}"
                        logger.debug(f"Found color combo '{color_desc}' after {item_type}")
                        return color_desc
            
            # Fall back to None to signal legacy detection
            return None
        
        # Helper function to find color associated with a specific item
        def find_color_for_item(item_type: str, item_patterns: list, caption_lower: str, used_colors: list = None) -> str:
            """Find the color that appears closest to an item mention in the caption"""
            if used_colors is None:
                used_colors = []
            
            # FIRST: Try to extract actual color description (combinations, modifiers)
            color_desc = extract_color_description(item_type, item_patterns, caption_lower)
            if color_desc and color_desc != "unknown":
                return color_desc
            
            best_color = "unknown"
            min_distance = float('inf')
            best_match_score = 0
            
            # Find all positions where this item type is mentioned
            item_positions = []
            for pattern in item_patterns:
                for match in re.finditer(pattern, caption_lower):
                    item_positions.append(match.start())
            
            if not item_positions:
                return "unknown"
            
            # PRIORITY: Check for multi-color patterns FIRST (e.g., "black and white")
            # These should override single color matches
            multi_color_patterns = {
                "black and white": [r"\bblack\s+and\s+white\b", r"\bblack\s*&\s*white\b"],
                "blue and white": [r"\bblue\s+and\s+white\b"],
                "red and white": [r"\bred\s+and\s+white\b"],
                "black and red": [r"\bblack\s+and\s+red\b"],
                "blue and gray": [r"\bblue\s+and\s+gr[ae]y\b"],
                "navy and white": [r"\bnavy\s+and\s+white\b"],
            }
            for color_name, color_patterns_list in multi_color_patterns.items():
                for color_pattern in color_patterns_list:
                    if re.search(color_pattern, caption_lower):
                        # Check if this multi-color is mentioned with/near this item
                        for item_pattern in item_patterns:
                            # Look for patterns like "black and white sweater"
                            pattern1 = rf"{color_pattern}\s+{item_pattern}"
                            pattern2 = rf"{item_pattern}\s+{color_pattern}"
                            pattern3 = rf"{color_pattern}.*{item_pattern}"  # Any distance
                            if (re.search(pattern1, caption_lower) or 
                                re.search(pattern2, caption_lower) or
                                re.search(pattern3, caption_lower)):
                                return color_name
            
            # First, try to find direct color-item pairs like "white t-shirt" or "white shirt"
            for color_name, color_patterns in colors.items():
                # Skip colors that are already assigned to other items (unless it's white/black which can appear multiple times)
                if color_name in used_colors and color_name not in ['white', 'black']:
                    continue
                    
                for color_pattern in color_patterns:
                    for item_pattern in item_patterns:
                        # Look for patterns like "white t-shirt", "white shirt", "t-shirt white", etc.
                        # Use word boundaries to avoid partial matches
                        pattern1 = rf"\b{color_pattern}\s+{item_pattern}\b"
                        pattern2 = rf"\b{item_pattern}\s+{color_pattern}\b"
                        if re.search(pattern1, caption_lower) or re.search(pattern2, caption_lower):
                            return color_name
            
            # If no direct pair found, look for colors near item mentions
            for item_pos in item_positions:
                # Look in a window of 30 characters before and after the item (smaller window for better accuracy)
                window_start = max(0, item_pos - 30)
                window_end = min(len(caption_lower), item_pos + 30)
                window_text = caption_lower[window_start:window_end]
                
                # Check each color pattern in this window
                for color_name, color_patterns in colors.items():
                    # Prefer white/black for shirts/t-shirts if they appear in the caption
                    if item_type in ['shirt', 't-shirt'] and color_name in ['white', 'black']:
                        # Check if this color appears in the caption at all
                        for color_pattern in color_patterns:
                            if re.search(rf"\b{color_pattern}\b", caption_lower):
                                # Give higher priority to white for inner layers
                                if color_name == 'white' and item_type in ['t-shirt', 'shirt']:
                                    return 'white'
                    
                    # Skip colors already used for other items (unless white/black)
                    if color_name in used_colors and color_name not in ['white', 'black']:
                        continue
                        
                    for color_pattern in color_patterns:
                        for color_match in re.finditer(rf"\b{color_pattern}\b", window_text):
                            # Calculate distance from item to color
                            color_pos = window_start + color_match.start()
                            distance = abs(item_pos - color_pos)
                            
                            # Score: closer is better, and colors before the item are slightly preferred
                            score = 1000 / (distance + 1)
                            if item_pos > color_pos:
                                score *= 1.2  # Prefer colors that come before the item
                            
                            if score > best_match_score:
                                best_match_score = score
                                min_distance = distance
                                best_color = color_name
            
            return best_color if best_color != "unknown" else "unknown"
        
        # Helper function to find texture/material for an item
        def find_texture_for_item(item_type: str, caption_lower: str) -> str:
            """Find texture/material ONLY if appropriate for the item type
            
            IMPORTANT: Only apply textures that make sense for specific items:
            - knit/wool: ONLY for sweaters, cardigans
            - denim: ONLY for jeans
            - cotton: for shirts, t-shirts
            - leather: for jackets, belts, bags
            - DO NOT apply textures to watches, glasses, or most accessories
            """
            # Items that should NOT have textile textures
            no_texture_items = ["watch", "glasses", "sunglasses", "bag", "hat", "cap"]
            if item_type in no_texture_items:
                return "unknown"
            
            # Only apply knit/wool to sweaters
            if item_type == "sweater":
                if re.search(r"\bknit\b|\bknitted\b|\bwool\b|\bcashmere\b", caption_lower):
                    return "knit"
                return "unknown"  # Don't default - let it be unknown
            
            # Only apply denim to jeans
            if item_type == "jeans":
                return "denim"  # Jeans are always denim
            
            # Only apply leather to specific items
            if item_type in ["jacket", "belt", "bag"]:
                if re.search(r"\bleather\b", caption_lower):
                    return "leather"
            
            # For pants - don't apply texture unless explicitly mentioned
            if item_type == "pants":
                if re.search(r"\bcotton\b", caption_lower):
                    return "cotton"
                if re.search(r"\bwool\b", caption_lower):
                    return "wool"
                return "unknown"  # Most pants don't need texture in search
            
            # For shirts/t-shirts
            if item_type in ["shirt", "t-shirt"]:
                if re.search(r"\bcotton\b", caption_lower):
                    return "cotton"
                if re.search(r"\bsilk\b", caption_lower):
                    return "silk"
                if re.search(r"\blinen\b", caption_lower):
                    return "linen"
                return "unknown"  # Don't force cotton
            
            return "unknown"
        
        # Special handling for accessory colors (watches often have metallic colors)
        def get_accessory_color(item_type: str, caption_lower: str, detected_colors: list) -> str:
            """Get color for accessories - check for color mentioned NEAR the item"""
            accessory_types = ["watch", "glasses", "jewelry", "bracelet", "ring", "belt"]
            
            if item_type in accessory_types:
                # Check for color mentioned WITH the item (e.g., "silver watch", "black belt")
                color_item_patterns = [
                    rf"\b(silver|metallic|steel|chrome|white|black|gold|golden|brown|leather)\s+{item_type}\b",
                    rf"\b{item_type}\s+(silver|metallic|steel|chrome|white|black|gold|golden|brown|leather)\b",
                ]
                
                for pattern in color_item_patterns:
                    match = re.search(pattern, caption_lower)
                    if match:
                        color_word = match.group(1) if match.group(1) else match.group(2)
                        # Map color words to standard colors
                        if color_word in ["silver", "metallic", "steel", "chrome"]:
                            return "silver"
                        elif color_word in ["gold", "golden"]:
                            return "gold"
                        elif color_word == "rose gold":
                            return "rose gold"
                        elif color_word in ["white", "black", "brown"]:
                            return color_word
                        elif color_word == "leather":
                            # Leather belts are often black or brown - check context
                            if re.search(r"\bblack\s+belt|\bbelt\s+black", caption_lower):
                                return "black"
                            elif re.search(r"\bbrown\s+belt|\bbelt\s+brown", caption_lower):
                                return "brown"
                            return "black"  # Default for leather
                
                # For watches, check for metallic colors anywhere (prioritize silver over white)
                if item_type == "watch":
                    # Check if silver/metallic is mentioned (even if not directly with "watch")
                    # Also check for "silver buckle" which often indicates silver watch
                    has_silver = re.search(r"\b(silver|metallic|steel|chrome)\b", caption_lower)
                    has_silver_buckle = re.search(r"\bsilver\s+buckle|\bbuckle\s+silver", caption_lower)
                    
                    if has_silver or has_silver_buckle:
                        # Only return white if "white watch" is EXPLICITLY mentioned
                        # Otherwise, prioritize silver (most watches are silver/metallic)
                        if not re.search(r"\bwhite\s+watch|\bwatch\s+white", caption_lower):
                            return "silver"
                    if re.search(r"\b(gold|golden)\b", caption_lower):
                        return "gold"
                    # If "white watch" is explicitly mentioned, return white
                    if re.search(r"\bwhite\s+watch|\bwatch\s+white", caption_lower):
                        return "white"
                    # Don't default to white - let normal detection handle it
                    
            return None  # Let normal color detection handle it
        
        # Find all colors in caption (for fallback)
        detected_colors = []
        for color_name, patterns in colors.items():
            for pattern in patterns:
                if re.search(pattern, caption_lower):
                    detected_colors.append(color_name)
                    break
        
        # Find all clothing items with proper color matching
        # IMPORTANT: Check for t-shirt patterns FIRST (more specific) before shirt (more general)
        # This prevents "t-shirt" from being misclassified as "shirt"
        
        # Track used colors to avoid conflicts
        used_colors = []
        
        # First pass: Check for t-shirt specifically (must come before shirt check)
        # BUT: Only detect t-shirt if there's NO collar mentioned (collars indicate a shirt, not t-shirt)
        tshirt_patterns = [r"\bt-shirt\b", r"\btee\b", r"\btshirt\b", r"\bt\s*shirt\b", r"\bcrew-neck\b", r"\bcrew\s*neck\b"]
        tshirt_found = False
        
        # Collar patterns indicate a SHIRT, not a t-shirt - check first
        collar_indicators = [
            r"\bcollar\b", r"\bcollared\b", r"\bdress\s*shirt\b", 
            r"\bbutton[\s-]*down\b", r"\bbutton[\s-]*up\b",
            r"\bpointed\s+collar\b", r"\bspread\s+collar\b"
        ]
        has_collar_in_caption = any(re.search(p, caption_lower) for p in collar_indicators)
        
        # Only check for t-shirt if NO collar is detected
        # If collar is detected, skip t-shirt detection entirely - it will be detected as a shirt
        if not has_collar_in_caption:
            for pattern in tshirt_patterns:
                if re.search(pattern, caption_lower):
                    if "t-shirt" not in found_items:
                        found_items.add("t-shirt")
                        # Check if there's a white/light color mentioned (common for inner layers)
                        item_color = find_color_for_item("t-shirt", tshirt_patterns, caption_lower, used_colors)
                        
                        # Special handling: if we detect a jacket/coat, the inner layer is likely white/light
                        if item_color == "unknown" and any(x in caption_lower for x in ["jacket", "coat", "blazer"]):
                            # Look specifically for white in the caption
                            if re.search(r"\bwhite\b", caption_lower):
                                item_color = "white"
                            elif re.search(r"\blight\b", caption_lower):
                                item_color = "white"  # Light colors often mean white/cream
                        
                        if item_color == "unknown" and detected_colors:
                            # Prefer white for t-shirts if it's in detected colors
                            if "white" in detected_colors:
                                item_color = "white"
                            else:
                                item_color = detected_colors[0]  # Fallback to first detected color
                        
                        if item_color != "unknown":
                            used_colors.append(item_color)
                        
                        # Detect texture/material for t-shirt
                        item_texture = find_texture_for_item("t-shirt", caption_lower)
                        
                        # Detect pattern (striped, plaid, etc.)
                        item_pattern = find_pattern_for_item("t-shirt", caption_lower)
                        
                        result["items"].append({
                            "type": "t-shirt",
                            "color": item_color,
                            "pattern": item_pattern,
                            "style": "casual",
                            "material": item_texture,
                            "features": [],
                            "description": caption
                        })
                        tshirt_found = True
                        logger.debug(f"Detected t-shirt (no collar): color={item_color}, pattern={item_pattern}, texture={item_texture}")
                    break
        else:
            logger.debug(f"Skipping t-shirt detection - collar detected in caption (will be detected as shirt)")
        
        # Second pass: Check for other clothing items (excluding t-shirt patterns)
        for item_type, patterns in clothing_patterns.items():
            if item_type == "t-shirt":
                continue  # Already handled above
            
            # For "shirt", make sure it doesn't match if "t-shirt" was already found
            # BUT: if the caption mentions "shirt" separately (not t-shirt), it might be a different item
            if item_type == "shirt" and "t-shirt" in found_items:
                # Only skip if we're confident it's the same item
                # Check if there's a separate mention of "shirt" (not part of "t-shirt")
                shirt_mentions = [m for m in re.finditer(r"\bshirt\b", caption_lower) 
                                 if not (m.start() > 0 and caption_lower[m.start()-2:m.start()].lower() in ['t-', 't ', 'tee'])]
                if not shirt_mentions:
                    # Skip shirt detection if t-shirt was already detected and no separate shirt mention
                    continue
            
            # IMPORTANT: If sweater is detected, skip jacket (they're often the same garment)
            # Cardigans and knit items are sweaters, not jackets
            if item_type == "jacket" and "sweater" in found_items:
                logger.debug(f"Skipping jacket detection - sweater already found (likely same garment)")
                continue
            
            # Also skip sweater if jacket already found with same color context
            if item_type == "sweater" and "jacket" in found_items:
                # Check if it's likely the same garment (knit/cardigan would be sweater)
                if any(x in caption_lower for x in ["knit", "cardigan", "pullover", "wool"]):
                    # It's more likely a sweater - remove jacket and add sweater instead
                    logger.debug(f"Correcting: jacket -> sweater (knit/cardigan detected)")
                    # Find and update the jacket item to sweater
                    for i, item in enumerate(result["items"]):
                        if item.get("type") == "jacket":
                            result["items"][i]["type"] = "sweater"
                            found_items.discard("jacket")
                            found_items.add("sweater")
                            break
                    continue
                else:
                    logger.debug(f"Skipping sweater detection - jacket already found")
                    continue
            
            # STRICT: Only detect shirt if collar is explicitly visible (not just mentioned)
            if item_type == "shirt" and "shirt" not in found_items and "t-shirt" not in found_items:
                has_sweater = any(x in caption_lower for x in ["sweater", "cardigan", "pullover"])
                
                # Check the new layered clothing prompts
                # "under the sweater there is a" and "the collar visible is"
                under_sweater_segment = None
                collar_segment = None
                for segment in caption_lower.split(" | "):
                    if "under the sweater" in segment or "under the" in segment:
                        under_sweater_segment = segment
                    if "collar" in segment and "visible" in segment:
                        collar_segment = segment
                
                # IMPORTANT: Check if it's a T-SHIRT, not a formal shirt
                # T-shirt should be detected as t-shirt, not shirt
                is_tshirt_mentioned = under_sweater_segment and re.search(r"\bt[\s-]*shirt\b", under_sweater_segment)
                if is_tshirt_mentioned:
                    # This is a t-shirt, not a formal shirt - skip shirt detection
                    # T-shirt will be detected separately
                    logger.debug(f"Skipping shirt detection - t-shirt mentioned in prompt (will be detected as t-shirt)")
                    # Force add t-shirt here since it was mentioned in the prompt
                    if "t-shirt" not in found_items:
                        found_items.add("t-shirt")
                        item_color = "white"
                        if "white" in under_sweater_segment:
                            item_color = "white"
                        elif "black" in under_sweater_segment:
                            item_color = "black"
                        result["items"].append({
                            "type": "t-shirt",
                            "color": item_color,
                            "pattern": "solid",
                            "style": "casual",
                            "material": "cotton",
                            "features": [],
                            "description": "T-shirt detected from layered clothing prompt"
                        })
                        logger.debug(f"Added t-shirt to items: color={item_color}")
                    continue
                
                # Check if shirt is mentioned in layered clothing prompts (formal shirt, not t-shirt)
                shirt_under_sweater = under_sweater_segment and re.search(r"\bshirt\b|\bblouse\b", under_sweater_segment) and not is_tshirt_mentioned
                collar_visible_from_prompt = collar_segment and re.search(r"\bwhite\b|\bcollar\b|\bshirt\b", collar_segment)
                
                # Require explicit collar visibility patterns (strong indicators)
                collar_visible_patterns = [
                    r"\bwhite\s+collar\b",  # White collar is a strong indicator
                    r"\bcollar\s+visible\b",
                    r"\bcollar\s+showing\b",
                    r"\bvisible\s+collar\b",
                    r"\bcollar\s+peeking\b",
                    r"\bcollared\s+shirt\b",  # Explicit collared shirt
                    r"\bshirt\s+with\s+collar\b",  # Shirt with collar
                    r"\bwearing\s+a\s+shirt\b",  # Explicitly wearing a shirt
                    r"\bunder\s+(?:the\s+)?sweater.*shirt\b",  # Shirt under sweater
                    r"\bshirt\s+under\b",  # Shirt under something
                ]
                has_visible_collar = any(re.search(p, caption_lower) for p in collar_visible_patterns)
                
                # Detect shirt if:
                # 1. Shirt mentioned in "under the sweater" prompt, OR
                # 2. Collar visible AND it's clearly a separate garment (not just describing the sweater)
                # 3. Has sweater AND collar explicitly visible as a SEPARATE item
                shirt_should_be_added = False
                
                # Check if the collar prompt is just describing the sweater (false positive)
                collar_describes_sweater = collar_segment and any(word in collar_segment for word in ["sweater", "pullover", "jumper", "cardigan"])
                
                if shirt_under_sweater:
                    logger.debug(f"Shirt detected (from 'under sweater' prompt: '{under_sweater_segment}')")
                    shirt_should_be_added = True
                elif collar_visible_from_prompt and not collar_describes_sweater:
                    # Only count as shirt if collar prompt doesn't just describe the sweater
                    logger.debug(f"Shirt detected (collar visible: '{collar_segment}')")
                    shirt_should_be_added = True
                elif has_sweater and has_visible_collar and not collar_describes_sweater:
                    logger.debug(f"Shirt detected (collar explicitly visible under sweater)")
                    shirt_should_be_added = True
                
                if shirt_should_be_added:
                    # Force add shirt since we detected it from prompts
                    # Don't wait for pattern matching since "shirt" might not be in caption
                    found_items.add("shirt")
                    item_color = "white"  # Collared shirts under sweaters are typically white
                    
                    # Try to extract color from the prompt response
                    if under_sweater_segment:
                        if "white" in under_sweater_segment:
                            item_color = "white"
                        elif "blue" in under_sweater_segment:
                            item_color = "light blue"
                    if collar_segment:
                        if "white" in collar_segment:
                            item_color = "white"
                    
                    result["items"].append({
                        "type": "shirt",
                        "color": item_color,
                        "pattern": "solid",
                        "style": "formal",
                        "material": "unknown",
                        "features": ["collared"],
                        "description": "Collared shirt visible under sweater"
                    })
                    logger.debug(f"Added shirt to items: color={item_color}")
                    continue  # Skip the normal pattern matching for shirt
                else:
                    # Skip - collar not explicitly visible
                    logger.debug(f"Skipping shirt detection - no visible collar mention")
                    continue
            
            # IMPORTANT: Pants and Jeans are mutually exclusive - only detect one
            # PREFER jeans over pants when denim/blue is mentioned (more specific)
            # If jeans already detected, skip pants
            if item_type == "pants" and "jeans" in found_items:
                logger.debug(f"Skipping pants detection - jeans already found (same garment)")
                continue
            
            # For jeans: if pants was already detected, check if we should upgrade to jeans
            if item_type == "jeans":
                # Check for denim/blue/jeans indicators
                # 1. Check our dedicated jeans prompt: "the person is wearing jeans that are"
                # 2. Check natural caption segments
                
                jeans_prompt_segment = None
                for segment in caption_lower.split(" | "):
                    if "wearing jeans that are" in segment:
                        jeans_prompt_segment = segment
                        break
                
                # If jeans prompt returns a color (not just repeating "jeans"), it's likely jeans
                jeans_from_prompt = False
                jeans_color_from_prompt = None
                if jeans_prompt_segment:
                    # Check if it describes actual jeans (has color like blue, black, etc.)
                    jeans_color_match = re.search(r"jeans that are\s+(\w+)", jeans_prompt_segment)
                    if jeans_color_match:
                        jeans_color = jeans_color_match.group(1).lower()
                        # Valid jeans colors - ONLY colors that indicate actual denim jeans
                        # "white" jeans exist but are less common - could be white pants
                        # "blue" is the strongest indicator of actual jeans
                        if jeans_color in ["blue", "light", "dark", "faded", "washed", "indigo", "navy"]:
                            jeans_from_prompt = True
                            jeans_color_from_prompt = jeans_color
                            logger.debug(f"Jeans detected from prompt: '{jeans_prompt_segment}' (color: {jeans_color})")
                        elif jeans_color in ["black", "gray", "grey"]:
                            # Black/gray could be jeans OR pants - check if denim mentioned elsewhere
                            if re.search(r"\bdenim\b", caption_lower):
                                jeans_from_prompt = True
                                jeans_color_from_prompt = jeans_color
                                logger.debug(f"Jeans detected from prompt with denim confirmation: '{jeans_prompt_segment}'")
                        # Skip white - too often confused with white dress pants
                
                # Get segments that are NOT our jeans prompt
                non_jeans_prompt_segments = []
                for segment in caption_lower.split(" | "):
                    if "wearing jeans that are" not in segment:
                        non_jeans_prompt_segments.append(segment)
                
                natural_caption = " ".join(non_jeans_prompt_segments)
                
                # Check for denim/jeans indicators in natural captions (not prompted)
                has_denim_indicators = re.search(r"\bdenim\b|\bjeans\b|\bblue\s+jeans\b|\bjean\b", natural_caption)
                
                if "pants" in found_items:
                    # Pants already detected - check if this is actually jeans
                    if jeans_from_prompt or has_denim_indicators:
                        # Upgrade pants to jeans (more specific)
                        logger.debug(f"Upgrading pants to jeans (jeans indicators found)")
                        found_items.discard("pants")
                        # Find and remove the pants item from result
                        result["items"] = [item for item in result["items"] if item.get("type") != "pants"]
                        # Store the jeans color from prompt to use later
                        if jeans_color_from_prompt:
                            # We'll use this color for the jeans item
                            pass  # Color will be extracted in the normal flow but we override below
                    else:
                        logger.debug(f"Skipping jeans detection - pants already found and no jeans indicators")
                        continue
                elif not jeans_from_prompt and not has_denim_indicators:
                    # No pants detected yet, and no jeans indicators - skip
                    logger.debug(f"Skipping jeans detection - no jeans indicators found")
                    continue
            
            # STRICT: Only detect watch if it's EXPLICITLY described with details (not hallucinated)
            # We have specific watch prompts: "the watch on the wrist is" and "the person is wearing a watch that is"
            if item_type == "watch" and "watch" not in found_items:
                has_watch_mention = re.search(r"\bwatch\b", caption_lower)
                
                # Check the watch-specific prompt responses
                watch_segment = None
                for segment in caption_lower.split(" | "):
                    # Match our watch prompts
                    if ("watch on the wrist" in segment or 
                        "wearing a watch" in segment or
                        ("watch" in segment and (" is " in segment or "that is" in segment))):
                        watch_segment = segment
                        break
                
                # Watch is confirmed if:
                # 1. The watch prompt mentions a watch WITH actual details (color, band, etc.), OR
                # 2. Watch is mentioned with specific details elsewhere
                # IMPORTANT: Filter out hallucinated responses that don't describe a real watch
                
                # Check if watch segment actually describes a watch (not hallucinated)
                watch_segment_is_valid = False
                if watch_segment:
                    # Invalid/hallucinated patterns - these indicate BLIP is not describing a real watch
                    invalid_watch_patterns = [
                        r"best way to",  # "the best way to wear" - not describing a watch
                        r"on (his|her) face",  # Nonsense
                        r"sweater",  # Talking about sweater, not watch
                        r"shirt",  # Talking about shirt, not watch
                        r"pants",  # Talking about pants, not watch
                        r"striped",  # Describing clothing pattern, not watch
                    ]
                    
                    has_invalid_pattern = any(re.search(p, watch_segment) for p in invalid_watch_patterns)
                    
                    # Valid watch indicators - things that suggest a real watch is described
                    valid_watch_indicators = [
                        r"\b(silver|gold|leather|metal|digital|analog)\b",  # Materials
                        r"\bband\b",
                        r"\bstrap\b",
                        r"\bface\b",  # Watch face
                        r"\bdial\b",  # Watch dial
                        r"\bon\s+(his|her)\s+wrist\b",  # "on his/her wrist" = watch is there
                        r"\bwearing\s+a\s+watch\b",  # Explicit wearing
                    ]
                    
                    # Circular/nonsense patterns that indicate BLIP is confused
                    circular_patterns = [
                        r"attached to the watch",  # Circular nonsense
                    ]
                    has_circular = any(re.search(p, watch_segment) for p in circular_patterns)
                    has_valid_indicator = any(re.search(p, watch_segment) for p in valid_watch_indicators)
                    
                    # Valid if: has valid indicator AND no invalid clothing patterns AND no circular nonsense
                    watch_segment_is_valid = has_valid_indicator and not has_invalid_pattern and not has_circular
                
                watch_from_prompt = watch_segment and "watch" in watch_segment and watch_segment_is_valid
                
                # Watch detail patterns - additional patterns that confirm a real watch
                watch_detail_patterns = [
                    r"\bwrist\s*watch\b",  # Wristwatch explicit
                    r"\b(silver|gold|leather)\s+watch\b",  # Specific material + watch
                    r"\bwatch\s+with\s+(silver|gold|leather|metal)\s+band\b",  # Watch with band description
                    r"\bwatch\s+(band|strap|face|dial)\b",  # Watch parts mentioned
                    r"\bdigital\s+watch\b",
                    r"\banalog\s+watch\b",
                    r"\bwearing\s+a\s+watch\s+that\s+is\s+on\s+(his|her)\s+wrist\b",  # Full phrase
                    r"\bwatch\s+that\s+is\s+on\s+(his|her)\s+wrist\b",  # Watch on wrist
                ]
                has_watch_details = any(re.search(p, caption_lower) for p in watch_detail_patterns)
                
                if watch_from_prompt:
                    logger.debug(f"Watch detected (from watch prompt: '{watch_segment}')")
                elif has_watch_mention and has_watch_details:
                    logger.debug(f"Watch detected (explicit details found)")
                else:
                    # Skip - watch mentioned without details (likely hallucinated)
                    logger.debug(f"Skipping watch detection - no explicit details (likely hallucinated)")
                    continue
            
            # STRICT: Only detect belt if BOTH belt AND buckle are mentioned together (definitely visible)
            if item_type == "belt" and "belt" not in found_items:
                has_belt_mention = re.search(r"\bbelt\b", caption_lower)
                has_buckle = re.search(r"\bbuckle\b", caption_lower)
                
                # Check if belt is mentioned WITH a color (e.g., "black belt", "belt black")
                belt_color_patterns = [
                    r"\b(black|white|brown|leather|silver|gold)\s+belt\b",
                    r"\bbelt\s+(black|white|brown|leather|silver|gold)\b",
                ]
                belt_with_color = any(re.search(p, caption_lower) for p in belt_color_patterns)
                
                # Require BOTH belt AND (buckle near belt OR belt+color) - belt must be explicitly mentioned
                if has_belt_mention and (has_buckle or belt_with_color):
                    # Belt mentioned with buckle or color = definitely visible
                    if has_buckle and has_belt_mention:
                        # Check if they're close together (within 20 chars - stricter)
                        belt_pos = caption_lower.find("belt")
                        buckle_pos = caption_lower.find("buckle")
                        if abs(belt_pos - buckle_pos) < 20:
                            # Also check for patterns like "belt with buckle" or "buckle on belt"
                            belt_buckle_patterns = [
                                r"\bbelt.*buckle|buckle.*belt",
                                r"\bbelt\s+with\s+buckle",
                                r"\bbuckle\s+on\s+belt",
                            ]
                            if any(re.search(p, caption_lower) for p in belt_buckle_patterns):
                                logger.debug(f"Belt detected (belt and buckle mentioned together)")
                            else:
                                # They're close but not in same phrase - still allow if within 20 chars
                                logger.debug(f"Belt detected (belt and buckle close together)")
                        else:
                            # Belt and buckle too far apart - might be different items
                            logger.debug(f"Skipping belt - belt and buckle too far apart (>20 chars)")
                            continue
                    elif belt_with_color:
                        logger.debug(f"Belt detected (belt+color mentioned)")
                    else:
                        logger.debug(f"Skipping belt - belt mentioned but no buckle or color")
                        continue
                elif belt_with_color:
                    # Belt+color mentioned = visible (even without buckle)
                    logger.debug(f"Belt detected (belt+color mentioned)")
                else:
                    # Skip - belt not explicitly mentioned
                    logger.debug(f"Skipping belt detection - belt not mentioned with buckle or color")
                    continue
            
            # SMART: Only detect shoes if they are actually visible
            # Check the shoe prompt: "the shoes in this photo are" and feet visibility
            if item_type == "shoes" and "shoes" not in found_items:
                # Check our dedicated shoe prompt first
                shoe_prompt_segment = None
                for segment in caption_lower.split(" | "):
                    if "shoes in this photo" in segment:
                        shoe_prompt_segment = segment
                        break
                
                # Check if the shoe prompt describes actual shoes (not sweater, not generic)
                shoes_from_prompt = False
                shoe_prompt_rejected = False  # Track if shoe prompt was unreliable
                if shoe_prompt_segment:
                    # First, check if explicit footwear type is mentioned (sneakers, boots, heels, etc.)
                    explicit_footwear_types = [
                        r"\bsneakers\b", r"\bboots\b", r"\bheels\b", r"\bsandals\b",
                        r"\bloafers\b", r"\btrainers\b", r"\bflats\b", r"\bslippers\b",
                        r"\bpumps\b", r"\bmoccasins\b", r"\bwedges\b", r"\bmules\b",
                        r"\bplatforms\b", r"\boxfords\b", r"\bconverse\b",
                    ]
                    has_explicit_footwear = any(re.search(p, shoe_prompt_segment) for p in explicit_footwear_types)
                    
                    # If explicit footwear is mentioned, trust it even if other clothing is mentioned
                    # This handles cases like "white sneakers and a black jacket" - sneakers is valid!
                    if has_explicit_footwear:
                        shoes_from_prompt = True
                        logger.debug(f"Shoes detected from prompt (explicit footwear type): '{shoe_prompt_segment}'")
                    else:
                        # No explicit footwear type - check if OTHER clothing is mentioned (unreliable)
                        shoe_invalid_patterns = [
                            r"sweater", r"shirt", r"pants", r"striped", r"top", r"jacket",
                        ]
                        has_invalid = any(re.search(p, shoe_prompt_segment) for p in shoe_invalid_patterns)
                        
                        if has_invalid:
                            # If the shoe prompt mentions other clothing without explicit footwear, skip
                            shoe_prompt_rejected = True
                            logger.debug(f"Shoe prompt mentions other clothing (no footwear type), skipping: '{shoe_prompt_segment}'")
                        else:
                            # Valid shoe descriptions include colors, types
                            shoe_type_indicators = [
                                r"\b(white|black|brown|blue|red|gray|grey|pink|beige)\b",
                                r"\b(leather|canvas|running|athletic|casual|formal)\b",
                            ]
                            has_shoe_indicator = any(re.search(p, shoe_prompt_segment) for p in shoe_type_indicators)
                            
                            if has_shoe_indicator:
                                shoes_from_prompt = True
                                logger.debug(f"Shoes detected from prompt: '{shoe_prompt_segment}'")
                
                # If shoe prompt was explicitly rejected due to other clothing mentions, skip entirely
                if shoe_prompt_rejected:
                    continue
                
                feet_segment = None
                for segment in caption_lower.split(" | "):
                    if "feet" in segment or "foot" in segment:
                        feet_segment = segment
                        break
                
                # Check if feet are NOT visible (cropped, hidden, not shown)
                feet_not_visible_indicators = [
                    r"\bnot\s+visible\b", r"\bnot\s+shown\b", r"\bhidden\b",
                    r"\bcropped\b", r"\bcut\s+off\b", r"\bnot\s+in\s+frame\b",
                    r"\bcan't\s+see\b", r"\bcannot\s+see\b", r"\bnot\s+seen\b",
                    r"\bno\s+feet\b", r"\bnothing\b",
                ]
                
                # Check if actual footwear is mentioned ANYWHERE in the caption
                actual_footwear_patterns = [
                    r"\bshoes\b", r"\bsneakers\b", r"\bboots\b", r"\bheels\b",
                    r"\bsandals\b", r"\bloafers\b", r"\bflats\b", r"\bslippers\b",
                    r"\bfootwear\b", r"\btrainers\b", r"\bpumps\b", r"\bwedges\b",
                ]
                
                has_actual_footwear = any(re.search(p, caption_lower) for p in actual_footwear_patterns)
                
                # If shoes detected from our prompt, proceed
                if shoes_from_prompt:
                    logger.debug(f"Proceeding with shoe detection (from shoe prompt)")
                elif feet_segment:
                    feet_not_visible = any(re.search(p, feet_segment) for p in feet_not_visible_indicators)
                    
                    if feet_not_visible:
                        logger.debug(f"Skipping shoes detection - feet not visible ('{feet_segment}')")
                        continue
                    elif has_actual_footwear:
                        logger.debug(f"Feet are visible and footwear mentioned - proceeding with shoe detection")
                    else:
                        logger.debug(f"Skipping shoes detection - feet visible but no footwear described in caption")
                        continue
                else:
                    # No feet segment and no shoe prompt - only detect if footwear explicitly mentioned
                    if not has_actual_footwear:
                        logger.debug(f"Skipping shoes detection - no footwear mentioned and no feet visibility info")
                        continue
            
            for pattern in patterns:
                # Use cleaned caption (prompts stripped) for item TYPE detection
                # This prevents false positives from prompt echoes like "under the sweater there is a"
                if re.search(pattern, caption_cleaned):
                    if item_type not in found_items:
                        found_items.add(item_type)
                        
                        # Special handling for accessories (watches, jewelry, belt) - check colors first
                        accessory_color = get_accessory_color(item_type, caption_lower, detected_colors)
                        if accessory_color:
                            item_color = accessory_color
                        else:
                            # Find the color specifically associated with this item
                            item_color = find_color_for_item(item_type, patterns, caption_lower, used_colors)
                        
                        # Special handling for watch - prioritize silver when silver buckle is mentioned
                        if item_type == "watch" and item_color == "unknown":
                            # Check for silver buckle (often indicates silver watch)
                            if re.search(r"\bsilver\s+buckle|\bbuckle\s+silver", caption_lower):
                                item_color = "silver"
                                print(f"  Watch color set to silver (silver buckle detected)")
                            # Check for silver/metallic mentioned near watch
                            elif re.search(r"\b(silver|metallic|steel|chrome)\b", caption_lower):
                                # Check if it's near "watch" (within 30 chars)
                                silver_match = re.search(r"\b(silver|metallic|steel|chrome)\b", caption_lower)
                                watch_match = re.search(r"\bwatch\b", caption_lower)
                                if silver_match and watch_match:
                                    if abs(silver_match.start() - watch_match.start()) < 30:
                                        item_color = "silver"
                                        logger.debug(f"Watch color set to silver (silver near watch)")
                        
                        # For watch, avoid picking up "white" from other items (like "white pants")
                        if item_type == "watch" and item_color == "white":
                            # Only keep white if "white watch" is explicitly mentioned
                            if not re.search(r"\bwhite\s+watch|\bwatch\s+white", caption_lower):
                                # Check if silver is mentioned anywhere (prioritize silver)
                                if re.search(r"\b(silver|metallic|steel|chrome)\b", caption_lower):
                                    item_color = "silver"
                                    print(f"  Watch color changed from white to silver (silver mentioned)")
                                elif re.search(r"\bsilver\s+buckle|\bbuckle\s+silver", caption_lower):
                                    item_color = "silver"
                                    print(f"  Watch color changed from white to silver (silver buckle)")
                                else:
                                    # If no silver, keep white only if explicitly "white watch"
                                    item_color = "unknown"  # Let it be unknown rather than wrong
                                    logger.debug(f"Watch color reset to unknown (white not explicitly with watch)")
                        
                        # Special handling for belt - must have color mentioned with belt or buckle
                        if item_type == "belt" and item_color == "unknown":
                            # Check for color directly with belt
                            belt_color_match = re.search(r"\b(black|white|brown|leather|silver|gold)\s+belt\b|\bbelt\s+(black|white|brown|leather|silver|gold)\b", caption_lower)
                            if belt_color_match:
                                color_word = belt_color_match.group(1) if belt_color_match.group(1) else belt_color_match.group(2)
                                if color_word in ["black", "white", "brown"]:
                                    item_color = color_word
                                elif color_word == "leather":
                                    # Check if black or brown mentioned nearby
                                    if re.search(r"\bblack", caption_lower):
                                        item_color = "black"
                                    elif re.search(r"\bbrown", caption_lower):
                                        item_color = "brown"
                                    else:
                                        item_color = "black"  # Default for leather
                            else:
                                # Check for color near "buckle" (e.g., "black belt with silver buckle")
                                buckle_color_match = re.search(r"\b(black|white|brown)\s+belt.*buckle|buckle.*\b(black|white|brown)\s+belt", caption_lower)
                                if not buckle_color_match:
                                    # Check if color appears before "buckle" (e.g., "black belt with buckle")
                                    for color in ["black", "white", "brown"]:
                                        if re.search(rf"\b{color}\b.*buckle|buckle.*\b{color}\b", caption_lower):
                                            # Check if belt is mentioned nearby
                                            belt_pos = caption_lower.find("belt")
                                            color_pos = caption_lower.find(color)
                                            buckle_pos = caption_lower.find("buckle")
                                            if belt_pos != -1 and (abs(belt_pos - color_pos) < 20 or abs(buckle_pos - color_pos) < 20):
                                                item_color = color
                                                break
                                    
                                    # If still unknown but buckle is mentioned, check for "black" anywhere (most common belt color)
                                    if item_color == "unknown" and re.search(r"\bbuckle\b", caption_lower):
                                        if re.search(r"\bblack\b", caption_lower):
                                            item_color = "black"
                                        elif re.search(r"\bbrown\b", caption_lower):
                                            item_color = "brown"
                                        elif re.search(r"\bwhite\b", caption_lower):
                                            item_color = "white"
                                        else:
                                            # Default to black if buckle mentioned but no color found
                                            item_color = "black"
                        
                        # Special handling for shirt: if there's a jacket, the shirt is likely white/light
                        if item_type == "shirt" and item_color == "unknown":
                            if any(x in caption_lower for x in ["jacket", "coat", "blazer"]):
                                if re.search(r"\bwhite\b", caption_lower):
                                    item_color = "white"
                                elif re.search(r"\blight\b", caption_lower):
                                    item_color = "white"
                        
                        # Special handling for jeans - ALWAYS check jeans prompt for accurate color
                        # The pants prompt often gives wrong color for jeans
                        if item_type == "jeans":
                            # Look specifically at the jeans prompt response
                            jeans_segment = None
                            for segment in caption_lower.split(" | "):
                                if "jeans that are" in segment:
                                    jeans_segment = segment
                                    break
                            
                            if jeans_segment:
                                # Extract color from jeans-specific segment
                                # "the person is wearing jeans that are blue and white"
                                jeans_color_part = jeans_segment.split("jeans that are")[-1].strip()
                                
                                light_blue_match = re.search(r"\blight\s*blue\b", jeans_color_part)
                                blue_match = re.search(r"\bblue\b", jeans_color_part)
                                black_match = re.search(r"\bblack\b", jeans_color_part)
                                white_match = re.search(r"\bwhite\b", jeans_color_part)
                                gray_match = re.search(r"\bgr[ae]y\b", jeans_color_part)
                                
                                if light_blue_match:
                                    item_color = "light blue"
                                    logger.debug(f"Jeans color set to light blue (from jeans prompt)")
                                elif blue_match:
                                    item_color = "blue"
                                    logger.debug(f"Jeans color set to blue (from jeans prompt: '{jeans_color_part}')")
                                elif black_match:
                                    item_color = "black"
                                    logger.debug(f"Jeans color set to black (from jeans prompt)")
                                elif white_match:
                                    item_color = "white"
                                    logger.debug(f"Jeans color set to white (from jeans prompt)")
                                elif gray_match:
                                    item_color = "gray"
                                    logger.debug(f"Jeans color set to gray (from jeans prompt)")
                                else:
                                    # Default to blue for classic denim
                                    item_color = "blue"
                                    logger.debug(f"Jeans color defaulted to blue (classic denim)")
                            elif item_color == "unknown" or "and" in str(item_color):
                                # No jeans segment found and color is uncertain
                                if re.search(r"\bblack\s+jeans|\bjeans.*black", caption_lower):
                                    item_color = "black"
                                elif re.search(r"\bwhite\s+jeans|\bjeans.*white", caption_lower):
                                    item_color = "white"
                                else:
                                    item_color = "blue"  # Default
                                    logger.debug(f"Jeans color defaulted to blue (no jeans prompt found)")
                        
                        # Special handling for shoes - simplify color if it's complex
                        if item_type == "shoes":
                            # If color has "and" but shoes are typically single color
                            # Take the first/primary color
                            if " and " in str(item_color):
                                colors_in_desc = item_color.split(" and ")
                                # Prefer the more specific color
                                if "light blue" in colors_in_desc:
                                    item_color = "light blue"
                                elif "white" in colors_in_desc:
                                    item_color = "white"
                                elif "black" in colors_in_desc:
                                    item_color = "black"
                                else:
                                    item_color = colors_in_desc[0].strip()
                                logger.debug(f"Shoes color simplified to '{item_color}'")
                        
                        # Special handling for watch - default to silver if no color detected
                        if item_type == "watch" and item_color == "unknown":
                            # Check for silver buckle (often indicates silver watch)
                            if re.search(r"\bsilver\s+buckle|\bbuckle\s+silver", caption_lower):
                                item_color = "silver"
                                print(f"  Watch color defaulted to silver (silver buckle)")
                            # Check for any silver/metallic mention
                            elif re.search(r"\b(silver|metallic|steel|chrome)\b", caption_lower):
                                item_color = "silver"
                                print(f"  Watch color defaulted to silver (metallic mentioned)")
                            # Default to silver (most watches are silver/metallic)
                            else:
                                item_color = "silver"
                                print(f"  Watch color defaulted to silver (no color detected)")
                        
                        # For accessories, don't use fallback colors - they should have explicit colors
                        if item_color == "unknown" and item_type not in ["watch", "belt", "glasses"]:
                            # Fallback: use a color that hasn't been used yet
                            for color in detected_colors:
                                if color not in used_colors:
                                    item_color = color
                                    break
                            if item_color == "unknown":
                                item_color = detected_colors[0]  # Last resort
                        
                        if item_color != "unknown" and item_color not in used_colors:
                            used_colors.append(item_color)
                        
                        # Detect texture/material for this item
                        item_texture = find_texture_for_item(item_type, caption_lower)
                        
                        # DO NOT force default textures - let find_texture_for_item handle it
                        # Only add texture if explicitly detected by VLM
                        # This prevents "Triko" from being added to everything
                        
                        # For belt, require a color - but we have fallback to "black" if buckle mentioned
                        # Only skip if no buckle mentioned AND no color detected
                        if item_type == "belt" and item_color == "unknown":
                            has_buckle = re.search(r"\bbuckle\b", caption_lower)
                            if not has_buckle:
                                # No buckle and no color = belt not clearly visible
                                logger.debug(f"Skipping belt - no color and no buckle detected")
                                found_items.discard("belt")
                                break
                            else:
                                # Buckle mentioned but no color found - use black as default
                                item_color = "black"
                                print(f"  Belt color defaulted to black (buckle mentioned but no color found)")
                        
                        # Detect closure type (buttoned/zippered)
                        item_closure = find_closure_for_item(item_type, caption_lower)
                        item_features = []
                        if item_closure != "unknown":
                            item_features.append(item_closure)
                        
                        # Detect pattern (striped, plaid, etc.)
                        item_pattern = find_pattern_for_item(item_type, caption_lower)
                        
                        result["items"].append({
                            "type": item_type,
                            "color": item_color,
                            "pattern": item_pattern,
                            "style": "casual",
                            "material": item_texture,
                            "closure": item_closure,
                            "features": item_features,
                            "description": caption
                        })
                        logger.debug(f"Detected {item_type}: color={item_color}, pattern={item_pattern}, texture={item_texture}, closure={item_closure}")
                    break  # Found this item type, move to next
        
        # If still no items found, try to detect "man" or "woman" wearing something
        if not result["items"]:
            # Check if it's describing a person
            if any(word in caption_lower for word in ["man", "woman", "person", "wearing", "outfit", "clothing", "dressed"]):
                # Try to infer items from common outfit descriptions - be more aggressive
                # Look for any clothing-related keywords
                clothing_keywords = {
                    "top": ["top", "shirt", "blouse", "sweater", "hoodie", "jacket", "coat"],
                    "bottom": ["pants", "jeans", "trousers", "shorts", "skirt"],
                    "dress": ["dress", "gown"],
                    "shoes": ["shoes", "sneakers", "boots", "heels", "sandals"]
                }
                
                # Try to find at least one specific item
                found_any = False
                for category, keywords in clothing_keywords.items():
                    for keyword in keywords:
                        if keyword in caption_lower:
                            # Determine color
                            item_color = detected_colors[0] if detected_colors else "unknown"
                            
                            # Map to specific item type
                            if keyword in ["top", "shirt"]:
                                item_type = "shirt"
                            elif keyword in ["jeans", "trousers"]:
                                item_type = "jeans"
                            elif keyword in ["pants"]:
                                item_type = "pants"
                            elif keyword in ["dress", "gown"]:
                                item_type = "dress"
                            elif keyword in ["shoes", "sneakers", "boots"]:
                                item_type = "shoes"
                            else:
                                item_type = keyword
                            
                            result["items"].append({
                                "type": item_type,
                                "color": item_color,
                                "pattern": "solid",
                                "style": "casual",
                                "material": "unknown",
                                "features": [],
                                "description": caption
                            })
                            found_any = True
                            break
                    if found_any:
                        break
                
                # Only use "outfit" as absolute last resort - but mark it so we can filter it out
                if not result["items"]:
                    # Try one more time with very basic detection
                    if "dress" in caption_lower or "gown" in caption_lower:
                        result["items"].append({
                            "type": "dress",
                            "color": detected_colors[0] if detected_colors else "unknown",
                            "pattern": "solid",
                            "style": "casual",
                            "material": "unknown",
                            "features": [],
                            "description": caption
                        })
                    elif any(word in caption_lower for word in ["blouse"]):
                        # Only detect blouse explicitly - avoid generic "shirt" or "top" detection
                        result["items"].append({
                            "type": "blouse",
                            "color": detected_colors[0] if detected_colors else "unknown",
                            "pattern": "solid",
                            "style": "casual",
                            "material": "unknown",
                            "features": [],
                            "description": caption
                        })
                    elif any(word in caption_lower for word in ["pants", "jeans", "trousers"]):
                        result["items"].append({
                            "type": "jeans" if "jean" in caption_lower else "pants",
                            "color": detected_colors[0] if detected_colors else "blue" if "jean" in caption_lower else "unknown",
                            "pattern": "solid",
                            "style": "casual",
                            "material": "denim" if "jean" in caption_lower else "unknown",
                            "features": [],
                            "description": caption
                        })
                    # If no specific items detected, leave items list empty
                    # This will trigger a helpful error message to the user
        
        # Determine overall style based on items and caption
        if any(item["type"] in ["suit", "blazer", "dress"] for item in result["items"]):
            result["overall_style"] = "formal"
            result["occasion"] = "business/special event"
        elif any(item["type"] in ["hoodie", "sneakers", "shorts"] for item in result["items"]):
            result["overall_style"] = "sporty/casual"
            result["occasion"] = "everyday/athletic"
        elif any(item["type"] == "jeans" for item in result["items"]):
            result["overall_style"] = "casual"
            result["occasion"] = "everyday"
        
        # Style keywords in caption
        style_keywords = {
            "casual": ["casual", "everyday", "relaxed", "comfortable"],
            "formal": ["formal", "elegant", "sophisticated", "business"],
            "sporty": ["sporty", "athletic", "active", "gym"],
            "bohemian": ["bohemian", "boho", "flowy", "artistic"],
            "minimalist": ["minimalist", "simple", "clean", "basic"]
        }
        
        for style, keywords in style_keywords.items():
            if any(kw in caption_lower for kw in keywords):
                result["overall_style"] = style
                break
        
        # Log the final extracted items
        logger.info(f"👗 Extracted {len(result['items'])} fashion items:")
        for item in result["items"]:
            logger.info(f"   - {item['type']}: color={item['color']}, material={item.get('material', 'unknown')}")
        logger.info(f"   Overall style: {result['overall_style']}, Occasion: {result['occasion']}")
        
        return result
    
    def _extract_fallback_info(self, response: str) -> Dict:
        """
        Fallback method to extract fashion info when JSON parsing fails
        
        Args:
            response: Raw model response
            
        Returns:
            Basic fashion data dictionary
        """
        items = []
        
        # Try to extract item mentions
        item_keywords = ["dress", "shirt", "pants", "jacket", "coat", "skirt", 
                        "top", "blouse", "jeans", "shoes", "boots", "sneakers",
                        "bag", "handbag", "accessory", "jewelry", "watch"]
        
        response_lower = response.lower()
        found_items = [item for item in item_keywords if item in response_lower]
        
        for item_type in found_items[:MAX_CLOTHING_ITEMS]:
            items.append({
                "type": item_type,
                "color": "unknown",
                "pattern": "unknown",
                "style": "unknown",
                "material": "unknown",
                "features": [],
                "description": f"{item_type} found in image"
            })
        
        return {
            "items": items,
            "overall_style": "unknown",
            "occasion": "unknown",
            "raw_response": response[:500]  # Store first 500 chars for debugging
        }
    
    def get_search_queries(self, fashion_data: Dict) -> List[str]:
        """
        Generate search queries from fashion data for Trendyol search.
        Uses Turkish translations for better results on Trendyol.com
        
        Args:
            fashion_data: Dictionary with extracted fashion attributes
            
        Returns:
            List of search query strings (one per item)
        """
        queries = []
        items = fashion_data.get("items", [])
        gender = fashion_data.get("gender", "unknown")
        
        # Turkish translations for gender
        gender_turkish = {
            "male": "Erkek",
            "female": "Kadın",
            "unknown": ""
        }
        
        # Turkish translations for colors (base colors)
        color_turkish = {
            "white": "Beyaz",
            "black": "Siyah",
            "blue": "Mavi",
            "light blue": "Açık Mavi",
            "dark blue": "Koyu Mavi",
            "light": "Açık",
            "dark": "Koyu",
            "bright": "Parlak",
            "pale": "Soluk",
            "deep": "Koyu",
            "red": "Kırmızı",
            "green": "Yeşil",
            "yellow": "Sarı",
            "pink": "Pembe",
            "purple": "Mor",
            "orange": "Turuncu",
            "brown": "Kahverengi",
            "gray": "Gri",
            "grey": "Gri",
            "beige": "Bej",
            "cream": "Krem",
            "ivory": "Krem",
            "off-white": "Krem",
            "eggshell": "Krem",
            "ecru": "Krem",
            "navy": "Lacivert",
            "silver": "Gümüş",
            "gold": "Altın",
            "striped": "Çizgili",
            "olive": "Haki",
            "khaki": "Haki",
            "tan": "Ten Rengi",
            "maroon": "Bordo",
            "burgundy": "Bordo",
            "teal": "Petrol Mavisi",
            "coral": "Mercan",
            # Common combinations (pre-translated for accuracy)
            "black and white": "Siyah Beyaz",
            "blue and white": "Mavi Beyaz",
            "red and white": "Kırmızı Beyaz",
            "black and red": "Siyah Kırmızı",
            "blue and gray": "Mavi Gri",
            "navy and white": "Lacivert Beyaz",
        }
        
        def translate_color_to_turkish(color_str: str) -> str:
            """Translate any color description to Turkish, including combinations.
            
            Handles:
            - Single colors: "white" -> "Beyaz"
            - Color combinations: "black and white" -> "Siyah Beyaz"
            - Modified colors: "light blue" -> "Açık Mavi"
            - Dynamic combinations: "red and blue" -> "Kırmızı Mavi"
            """
            if not color_str or color_str == "unknown":
                return ""
            
            color_lower = color_str.lower().strip()
            
            # Check if it's a known combination first
            if color_lower in color_turkish:
                return color_turkish[color_lower]
            
            # Handle "X and Y" pattern
            if " and " in color_lower:
                parts = color_lower.split(" and ")
                translated_parts = []
                for part in parts:
                    part = part.strip()
                    if part in color_turkish:
                        translated_parts.append(color_turkish[part])
                    else:
                        # Try to translate modified colors (e.g., "light blue")
                        words = part.split()
                        if len(words) == 2 and words[0] in color_turkish and words[1] in color_turkish:
                            translated_parts.append(f"{color_turkish[words[0]]} {color_turkish[words[1]]}")
                        elif part:
                            translated_parts.append(part.capitalize())  # Keep original if unknown
                return " ".join(translated_parts)
            
            # Handle modified colors (e.g., "light blue", "dark green")
            words = color_lower.split()
            if len(words) == 2:
                modifier, base_color = words
                if modifier in color_turkish and base_color in color_turkish:
                    return f"{color_turkish[modifier]} {color_turkish[base_color]}"
            
            # Single word color
            if color_lower in color_turkish:
                return color_turkish[color_lower]
            
            # Unknown - return capitalized original
            return color_str.capitalize()
        
        # Turkish translations for clothing items
        item_turkish = {
            "sweater": "Kazak",
            "pants": "Pantolon",
            "jeans": "Kot Pantolon",
            "shirt": "Gömlek",
            "t-shirt": "Tişört",
            "blouse": "Bluz",
            "dress": "Elbise",
            "skirt": "Etek",
            "jacket": "Ceket",
            "coat": "Mont",
            "hoodie": "Sweatshirt",
            "shorts": "Şort",
            "belt": "Kemer",
            "watch": "Saat",
            "bag": "Çanta",
            "shoes": "Ayakkabı",
            "sneakers": "Spor Ayakkabı",
            "boots": "Bot",
            "hat": "Şapka",
            "scarf": "Atkı",
            "glasses": "Gözlük",
            "sunglasses": "Güneş Gözlüğü",
        }
        
        # Turkish translations for closures/features
        closure_turkish = {
            "zippered": "Fermuarlı",
            "buttoned": "Düğmeli",
        }
        
        # Turkish translations for materials
        material_turkish = {
            "knit": "Triko",
            "cotton": "Pamuklu",
            "denim": "Kot",
            "leather": "Deri",
            "wool": "Yün",
            "silk": "İpek",
        }
        
        # Turkish translations for patterns
        pattern_turkish = {
            "striped": "Çizgili",
            "plaid": "Kareli",
            "floral": "Çiçekli",
            "polka dot": "Puantiyeli",
            "solid": "",  # Don't add solid to query
        }
        
        # Generic item types that are too vague for search - filter them out
        generic_types = {"outfit", "clothing", "garment", "apparel", "wear"}
        
        logger.info(f"Generating search queries for {len(items)} items (gender: {gender})")
        
        for item in items:
            item_type = item.get("type", "").lower()
            
            # Skip generic item types - they're too vague for meaningful search
            if item_type in generic_types:
                logger.debug(f"Skipping generic item type '{item_type}' - too vague for search")
                continue
            
            # Skip items marked as generic
            if item.get("is_generic", False):
                logger.debug(f"Skipping generic item: {item_type}")
                continue
            
            # Build query in Turkish for Trendyol
            query_parts = []
            
            # Add gender in Turkish (for better Trendyol results)
            gender_tr = gender_turkish.get(gender, "")
            if gender_tr:
                query_parts.append(gender_tr)
            
            # Add color in Turkish if available (using dynamic translation)
            color = item.get("color", "unknown")
            color_tr = translate_color_to_turkish(color)
            if color_tr:
                query_parts.append(color_tr)
            
            # Add pattern in Turkish if not solid
            pattern = item.get("pattern", "solid").lower()
            pattern_tr = pattern_turkish.get(pattern, "")
            if pattern_tr:  # Only add if not empty (solid is empty)
                query_parts.append(pattern_tr)
            
            # Add closure type in Turkish if available
            closure = item.get("closure", "unknown")
            closure_tr = closure_turkish.get(closure, "")
            if closure_tr:
                query_parts.append(closure_tr)
            
            # Add material in Turkish if available
            # Skip material for jeans (already implicit in "Kot Pantolon")
            material = item.get("material", "unknown").lower()
            if item_type != "jeans":  # Avoid "Kot Kot Pantolon"
                material_tr = material_turkish.get(material, "")
                if material_tr:
                    query_parts.append(material_tr)
            
            # Add item type in Turkish (required)
            item_tr = item_turkish.get(item_type, item_type.title())
            if item_tr:
                query_parts.append(item_tr)
            
            # Only create query if we have item type and it's not generic
            if query_parts and item_type not in generic_types:
                query = " ".join(query_parts)
                queries.append(query)
                logger.info(f"  ✓ Query: '{query}' (EN: {gender} {color} {pattern} {item_type})")
        
        # Remove duplicates and return
        unique_queries = list(dict.fromkeys(queries))[:10]  # Allow up to 10 unique item queries, preserve order
        
        logger.info(f"Generated {len(unique_queries)} unique search queries")
        return unique_queries

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
            print("⚠ Local models require transformers. Install with:")
            print("   pip install transformers torch")
            print("Falling back to API...")
            self.use_api = True
            self._setup_api()
    
    def _load_local_model(self):
        """Load model locally - downloads once, then cached forever"""
        print(f"🔄 Loading model: {self.model_name}")
        print("   (First run downloads model, then it's cached for instant loading)")
        
        try:
            import torch
            
            # Load BLIP model
            if "blip" in self.model_name.lower():
                print("   Loading BLIP image captioning model...")
                self.local_processor = BlipProcessor.from_pretrained(self.model_name)
                self.local_model = BlipForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.model_type = "blip"
                print(f"✓ BLIP model loaded successfully on {self.device.upper()}")
            else:
                # Try generic Vision2Seq for other models
                print(f"   Loading {self.model_name}...")
                from transformers import AutoProcessor, AutoModelForVision2Seq
                self.local_processor = AutoProcessor.from_pretrained(self.model_name)
                self.local_model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
                self.model_type = "generic"
                print(f"✓ Model loaded successfully on {self.device.upper()}")
                
        except Exception as e:
            print(f"⚠ Could not load local model: {e}")
            print("   Falling back to API (may not work)...")
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
            print("🔄 Analyzing image with local model...")
            
            if self.model_type == "blip":
                # BLIP conditional image captioning - use text prompts to guide toward fashion
                # Generate multiple captions with different prompts for better coverage
                captions = []
                
                # List of fashion-focused prompts to guide BLIP
                # Includes prompts for gender, clothing, colors, textures/materials, closures, and accessories
                fashion_prompts = [
                    "this is a photo of a [man/woman/boy/girl]",        # Gender detection
                    "a photo of a person wearing",                       # General clothing
                    "the clothing colors are",                           # Focus on colors
                    "the fabric and texture includes",                   # Texture/material focus
                    "the shirt has a collar or is collarless",           # Collar detection (shirt vs t-shirt)
                    "the neckline is crew neck or collared",             # Neckline type detection
                    "the sweater or jacket has a zipper or buttons",     # Zipper vs buttons detection
                    "the closure is a zipper running down the front",    # Explicit zipper detection
                    "the accessories include",                           # Accessories like watches
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
                    print(f"  Prompt '{prompt}': {caption}")
                
                # Combine all captions for comprehensive extraction
                combined_caption = " | ".join(captions)
                print(f"✓ Combined fashion caption: {combined_caption[:200]}...")
                return combined_caption
            else:
                # Generic model
                inputs = self.local_processor(images=image, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    output = self.local_model.generate(**inputs, max_new_tokens=200)
                
                text = self.local_processor.decode(output[0], skip_special_tokens=True)
                print(f"✓ Generated text: {text[:100]}...")
                return text
                
        except Exception as e:
            print(f"⚠ Local model analysis failed: {e}")
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
                        print("Trying InferenceClient image_to_text method...")
                        result = self.client.image_to_text(image=image, model=self.model_name)
                        if result:
                            generated_text = result
                            print("✓ Successfully used InferenceClient image_to_text")
                    except Exception as e:
                        print(f"InferenceClient image_to_text failed: {e}")
                        print("Falling back to direct API calls...")
                
                # For LLaVA or if InferenceClient failed, use direct API calls
                if not generated_text:
                    if is_llava_model:
                        prompt = self._create_fashion_prompt()
                        print("Sending request to Hugging Face Inference API with custom prompt...")
                    else:
                        prompt = None
                        print("Sending request to Hugging Face Inference API for image captioning...")
                    
                    try:
                        generated_text = self._call_api_direct(image, prompt)
                    except requests.exceptions.Timeout:
                        print("Request timeout - the model might be taking longer than expected")
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
            print(f"Error analyzing image: {e}")
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
        
        # Detect gender from caption
        if re.search(r"\b(man|men|male|guy|boy|gentleman)\b", caption_lower):
            result["gender"] = "male"
        elif re.search(r"\b(woman|women|female|girl|lady)\b", caption_lower):
            result["gender"] = "female"
        else:
            # Try to infer from clothing items (dresses/skirts typically female, suits often male)
            if any(word in caption_lower for word in ["dress", "skirt", "heels"]):
                result["gender"] = "female"
            elif any(word in caption_lower for word in ["suit", "tie"]):
                result["gender"] = "male"
        
        # Comprehensive clothing detection
        # IMPORTANT: Order matters! More specific patterns should come first
        clothing_patterns = {
            # Tops - Shirt (collared) should be detected separately from T-shirt (no collar)
            # Shirt = has collar, buttons, dress shirt, button-down, button-up
            "shirt": [r"\bshirt\b", r"\bdress\s*shirt\b", r"\bbutton[\s-]*down\b", r"\bbutton[\s-]*up\b", r"\bcollared\b", r"\bcollar\b"],
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
            "shoes": [r"\bshoes\b", r"\bsneakers\b", r"\bboots\b", r"\bheels\b", r"\bsandals\b", r"\bloafers\b"],
            # Accessories
            "bag": [r"\bbag\b", r"\bpurse\b", r"\bbackpack\b", r"\bhandbag\b"],
            "hat": [r"\bhat\b", r"\bcap\b", r"\bbeanie\b"],
            "watch": [r"\bwatch\b"],
            "glasses": [r"\bglasses\b", r"\bsunglasses\b"],
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
            "blue": [r"\bblue\b", r"\bnavy\b", r"\bdenim\b"],
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
            """Find closure type (button/zipper) mentioned in relation to an item"""
            # Items that commonly have closures
            # NOTE: t-shirt is NOT included - t-shirts don't have buttons/zippers
            # shirt = collared dress shirt with buttons
            closure_items = ["shirt", "jacket", "coat", "hoodie", "cardigan", "blazer", "sweater"]
            
            # T-shirts NEVER have closures - return immediately
            if item_type == "t-shirt":
                return "unknown"
            
            if item_type not in closure_items:
                return "unknown"
            
            # PRIORITY 1: Check for ZIPPER patterns FIRST (more specific, often visible)
            # Zippers are more visually distinct and should be prioritized
            zipper_patterns = [
                r"\bzipper\b", r"\bzipped\b", r"\bzip[\s-]*up\b", r"\bzippered\b",
                r"\bzip\s+front\b", r"\bfront\s+zip\b", r"\bzipping\b",
                r"\bzipper\s+running\b", r"\bzip\s+closure\b"
            ]
            for pattern in zipper_patterns:
                if re.search(pattern, caption_lower):
                    print(f"    Detected ZIPPER closure for {item_type}")
                    return "zippered"
            
            # PRIORITY 2: Check for explicit BUTTON patterns
            button_patterns = [
                r"\bbutton[\s-]*up\b", r"\bbutton[\s-]*down\b", r"\bbuttoned\b",
                r"\bbuttons\b", r"\bbutton\s+front\b", r"\bfront\s+button\b"
            ]
            for pattern in button_patterns:
                if re.search(pattern, caption_lower):
                    print(f"    Detected BUTTON closure for {item_type}")
                    return "buttoned"
            
            # PRIORITY 3: For sweaters - check if it's explicitly a zip-up sweater/cardigan
            if item_type == "sweater":
                # Check for zip-up sweater patterns
                if re.search(r"\bzip[\s-]*up\s*(sweater|cardigan|knit)\b", caption_lower):
                    return "zippered"
                if re.search(r"\b(sweater|cardigan|knit)\s+with\s+zip\b", caption_lower):
                    return "zippered"
                # Only default to buttoned if "cardigan" is mentioned AND no zip patterns found
                if re.search(r"\bcardigan\b", caption_lower):
                    # Cardigans are typically buttoned, but can be zippered
                    # Only use this default if no explicit closure was found above
                    print(f"    Defaulting to BUTTONED for cardigan (no explicit closure found)")
                    return "buttoned"
            
            # PRIORITY 4: For jackets/hoodies - check context
            if item_type in ["hoodie", "jacket"]:
                # Most hoodies/casual jackets are zip-up
                if re.search(r"\bzip\b", caption_lower):
                    return "zippered"
            
            # PRIORITY 5: For coats - check for buttoned coat patterns
            if item_type == "coat":
                if re.search(r"\bbuttoned\s*coat\b|\bcoat\s+with\s+buttons\b", caption_lower):
                    return "buttoned"
            
            return "unknown"
        
        # Helper function to find color associated with a specific item
        def find_color_for_item(item_type: str, item_patterns: list, caption_lower: str, used_colors: list = None) -> str:
            """Find the color that appears closest to an item mention in the caption"""
            if used_colors is None:
                used_colors = []
            
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
            """Find texture/material mentioned in relation to an item"""
            for texture_name, texture_patterns in textures.items():
                for pattern in texture_patterns:
                    if re.search(pattern, caption_lower):
                        return texture_name
            return "unknown"
        
        # Special handling for accessory colors (watches often have metallic colors)
        def get_accessory_color(item_type: str, caption_lower: str, detected_colors: list) -> str:
            """Get color for accessories like watches - prefer metallic/white over black"""
            accessory_types = ["watch", "glasses", "jewelry", "bracelet", "ring"]
            
            if item_type in accessory_types:
                # Check for metallic colors first (common for watches)
                if re.search(r"\b(silver|metallic|steel|chrome)\b", caption_lower):
                    return "silver"
                if re.search(r"\b(gold|golden)\b", caption_lower):
                    return "gold"
                if re.search(r"\brose\s*gold\b", caption_lower):
                    return "rose gold"
                # Check for white (often used for watch dials/bands)
                if re.search(r"\bwhite\b", caption_lower):
                    return "white"
                # If nothing specific found, default to silver for watches (most common)
                if item_type == "watch":
                    return "silver"
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
        # Also check for "white" or "light" colors that might indicate an inner layer
        tshirt_patterns = [r"\bt-shirt\b", r"\btee\b", r"\btshirt\b", r"\bt\s*shirt\b", r"\bcrew-neck\b", r"\bcrew\s*neck\b"]
        tshirt_found = False
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
                    
                    result["items"].append({
                        "type": "t-shirt",
                        "color": item_color,
                        "pattern": "solid",
                        "style": "casual",
                        "material": item_texture,
                        "features": [],
                        "description": caption
                    })
                    tshirt_found = True
                    print(f"  Detected t-shirt: color={item_color}, texture={item_texture}")
                break
        
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
                print(f"  Skipping jacket detection - sweater already found (likely same garment)")
                continue
            
            # Also skip sweater if jacket already found with same color context
            if item_type == "sweater" and "jacket" in found_items:
                # Check if it's likely the same garment (knit/cardigan would be sweater)
                if any(x in caption_lower for x in ["knit", "cardigan", "pullover", "wool"]):
                    # It's more likely a sweater - remove jacket and add sweater instead
                    print(f"  Correcting: jacket -> sweater (knit/cardigan detected)")
                    # Find and update the jacket item to sweater
                    for i, item in enumerate(result["items"]):
                        if item.get("type") == "jacket":
                            result["items"][i]["type"] = "sweater"
                            found_items.discard("jacket")
                            found_items.add("sweater")
                            break
                    continue
                else:
                    print(f"  Skipping sweater detection - jacket already found")
                    continue
            
            # Special case: If "shirt" is detected but no "t-shirt", and there's a sweater/cardigan,
            # Need to determine if inner layer is T-SHIRT (crew neck) or SHIRT (collared)
            if item_type == "shirt" and "t-shirt" not in found_items and "shirt" not in found_items:
                # Check if there's a sweater/cardigan (indicating layered outfit)
                has_cardigan_sweater = any(x in caption_lower for x in ["sweater", "cardigan", "pullover", "knit"])
                has_jacket_coat = any(x in caption_lower for x in ["jacket", "coat", "blazer"])
                
                if has_cardigan_sweater or has_jacket_coat:
                    # CRITICAL: Detect if the inner layer has a COLLAR or not
                    # Collar patterns indicate a SHIRT, not a t-shirt
                    collar_patterns = [
                        r"\bcollar\b", r"\bcollared\b", r"\bcollar\s+shirt\b",
                        r"\bdress\s*shirt\b", r"\bbutton[\s-]*up\b", r"\bbutton[\s-]*down\b",
                        r"\bpointed\s+collar\b", r"\bspread\s+collar\b", r"\bwhite\s+shirt\b",
                        r"\bshirt\s+collar\b", r"\bshirt\s+with\s+collar\b"
                    ]
                    
                    # T-shirt/crew neck patterns indicate a T-SHIRT
                    tshirt_patterns = [
                        r"\bcrew[\s-]*neck\b", r"\bround[\s-]*neck\b", r"\bcollarless\b",
                        r"\bt-shirt\b", r"\btee\b", r"\bno\s+collar\b"
                    ]
                    
                    has_collar = any(re.search(p, caption_lower) for p in collar_patterns)
                    has_crew_neck = any(re.search(p, caption_lower) for p in tshirt_patterns)
                    
                    print(f"    Inner layer detection: has_collar={has_collar}, has_crew_neck={has_crew_neck}")
                    
                    # If collar is detected OR no explicit crew neck -> prefer SHIRT
                    # If crew neck is detected AND no collar -> T-SHIRT
                    if has_collar or (not has_crew_neck and re.search(r"\bshirt\b", caption_lower) and not re.search(r"\bt[\s-]*shirt\b", caption_lower)):
                        # It's a COLLARED SHIRT
                        found_items.add("shirt")
                        shirt_color = "white"  # Default for inner layer
                        if re.search(r"\bwhite\b", caption_lower):
                            shirt_color = "white"
                        elif re.search(r"\blight\s*blue\b", caption_lower):
                            shirt_color = "light blue"
                        
                        item_texture = find_texture_for_item("shirt", caption_lower)
                        shirt_closure = find_closure_for_item("shirt", caption_lower)
                        if shirt_closure == "unknown":
                            shirt_closure = "buttoned"  # Collared shirts typically have buttons
                        
                        result["items"].append({
                            "type": "shirt",
                            "color": shirt_color,
                            "pattern": "solid",
                            "style": "casual",
                            "material": item_texture if item_texture != "unknown" else "cotton",
                            "closure": shirt_closure,
                            "features": ["collared", shirt_closure],
                            "description": caption
                        })
                        used_colors.append(shirt_color)
                        print(f"  Detected COLLARED SHIRT (under outerwear): color={shirt_color}, closure={shirt_closure}")
                        continue
                    
                    elif has_crew_neck or not has_collar:
                        # It's a t-shirt under the cardigan/sweater
                        found_items.add("t-shirt")
                        # Default to white for inner layer t-shirts
                        tshirt_color = "white"
                        if re.search(r"\bwhite\b", caption_lower):
                            tshirt_color = "white"
                        elif re.search(r"\bblack\b", caption_lower) and "white" not in detected_colors:
                            tshirt_color = "black"
                        
                        item_texture = find_texture_for_item("t-shirt", caption_lower)
                        result["items"].append({
                            "type": "t-shirt",
                            "color": tshirt_color,
                            "pattern": "solid",
                            "style": "casual",
                            "material": item_texture if item_texture != "unknown" else "cotton",
                            "features": [],  # T-shirts have NO buttons/closure
                            "description": caption
                        })
                        used_colors.append(tshirt_color)
                        print(f"  Detected T-SHIRT (under outerwear): color={tshirt_color}, texture={item_texture}")
                        continue
            
            for pattern in patterns:
                if re.search(pattern, caption_lower):
                    if item_type not in found_items:
                        found_items.add(item_type)
                        
                        # Special handling for accessories (watches, jewelry) - check metallic colors first
                        accessory_color = get_accessory_color(item_type, caption_lower, detected_colors)
                        if accessory_color:
                            item_color = accessory_color
                        else:
                            # Find the color specifically associated with this item
                            item_color = find_color_for_item(item_type, patterns, caption_lower, used_colors)
                        
                        # Special handling for shirt: if there's a jacket, the shirt is likely white/light
                        if item_type == "shirt" and item_color == "unknown":
                            if any(x in caption_lower for x in ["jacket", "coat", "blazer"]):
                                if re.search(r"\bwhite\b", caption_lower):
                                    item_color = "white"
                                elif re.search(r"\blight\b", caption_lower):
                                    item_color = "white"
                        
                        if item_color == "unknown" and detected_colors:
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
                        
                        # Default textures for common items if not detected
                        if item_texture == "unknown":
                            if item_type == "sweater":
                                item_texture = "knit"  # Sweaters are typically knit
                            elif item_type == "jeans":
                                item_texture = "denim"
                        
                        # Detect closure type (buttoned/zippered)
                        item_closure = find_closure_for_item(item_type, caption_lower)
                        item_features = []
                        if item_closure != "unknown":
                            item_features.append(item_closure)
                        
                        result["items"].append({
                            "type": item_type,
                            "color": item_color,
                            "pattern": "solid",
                            "style": "casual",
                            "material": item_texture,
                            "closure": item_closure,
                            "features": item_features,
                            "description": caption
                        })
                        print(f"  Detected {item_type}: color={item_color}, texture={item_texture}, closure={item_closure}")
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
                    elif any(word in caption_lower for word in ["shirt", "top", "blouse"]):
                        result["items"].append({
                            "type": "shirt",
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
        Uses simplified queries (gender + color + item type) for better results.
        Filters out generic "outfit" items to prevent vague searches.
        
        Args:
            fashion_data: Dictionary with extracted fashion attributes
            
        Returns:
            List of search query strings (one per item)
        """
        queries = []
        items = fashion_data.get("items", [])
        gender = fashion_data.get("gender", "unknown")
        
        # Generic item types that are too vague for search - filter them out
        generic_types = {"outfit", "clothing", "garment", "apparel", "wear"}
        
        for item in items:
            item_type = item.get("type", "").lower()
            
            # Skip generic item types - they're too vague for meaningful search
            if item_type in generic_types:
                print(f"Skipping generic item type '{item_type}' - too vague for search")
                continue
            
            # Skip items marked as generic
            if item.get("is_generic", False):
                print(f"Skipping generic item: {item_type}")
                continue
            
            # Build query: gender + color + closure + texture/material + item type
            # This gives better, gender-specific results on Trendyol
            query_parts = []
            
            # Add gender if detected (for better Trendyol results)
            if gender != "unknown":
                query_parts.append(gender)
            
            # Add color if available
            if item.get("color") and item["color"] != "unknown":
                query_parts.append(item["color"])
            
            # Add closure type if available (e.g., "buttoned shirt", "zip-up jacket")
            closure = item.get("closure", "unknown")
            if closure and closure != "unknown":
                # Convert closure to search-friendly term
                if closure == "buttoned":
                    query_parts.append("buttoned")
                elif closure == "zippered":
                    query_parts.append("zip-up")
            
            # Add texture/material if available (e.g., "knit sweater", "leather jacket")
            if item.get("material") and item["material"] != "unknown":
                query_parts.append(item["material"])
            
            # Add item type (required)
            if item_type:
                query_parts.append(item_type)
            
            # Only create query if we have item type and it's not generic
            if query_parts and item_type not in generic_types:
                query = " ".join(query_parts)
                queries.append(query)
                # Log detailed query breakdown
                parts_info = []
                if gender != "unknown":
                    parts_info.append(f"gender={gender}")
                if item.get("color") and item["color"] != "unknown":
                    parts_info.append(f"color={item['color']}")
                if closure and closure != "unknown":
                    parts_info.append(f"closure={closure}")
                if item.get("material") and item["material"] != "unknown":
                    parts_info.append(f"texture={item['material']}")
                parts_info.append(f"type={item_type}")
                print(f"  ✓ Query: '{query}' ({', '.join(parts_info)})")
        
        # Remove duplicates and return
        unique_queries = list(set(queries))[:10]  # Allow up to 10 unique item queries
        
        return unique_queries

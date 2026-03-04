"""
Vision Language Model Service for Fashion Analysis

Uses Qwen2-VL-7B-Instruct via Hugging Face Serverless Inference API:
- Open-access model (no gated model approval required)
- True multimodal LLM: reasons about images, not just captions
- Single structured prompt → clean JSON output
- No local model download needed (HF Spaces or local with HF_API_TOKEN)
"""

import re
import json
import base64
import time
import logging
import os
import sys
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

_file_handler = logging.FileHandler(os.path.join(log_dir, 'vlm_service.log'))
_file_handler.setLevel(logging.DEBUG)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s - %(levelname)s - [VLM] %(message)s')
_file_handler.setFormatter(_fmt)
_console_handler.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_file_handler)
    logger.addHandler(_console_handler)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import VLM_MODEL_NAME, HF_API_TOKEN

# ---------------------------------------------------------------------------
# Turkish Translation Tables
# ---------------------------------------------------------------------------

COLOR_TRANSLATIONS: Dict[str, str] = {
    "black": "Siyah",
    "white": "Beyaz",
    "gray": "Gri",
    "grey": "Gri",
    "navy": "Lacivert",
    "navy blue": "Lacivert",
    "blue": "Mavi",
    "light blue": "Açık Mavi",
    "dark blue": "Koyu Mavi",
    "royal blue": "Saks Mavi",
    "red": "Kırmızı",
    "dark red": "Koyu Kırmızı",
    "burgundy": "Bordo",
    "maroon": "Bordo",
    "wine": "Bordo",
    "green": "Yeşil",
    "dark green": "Koyu Yeşil",
    "olive": "Haki",
    "olive green": "Haki",
    "khaki": "Haki",
    "mint": "Mint",
    "mint green": "Mint",
    "brown": "Kahverengi",
    "dark brown": "Koyu Kahverengi",
    "tan": "Camel",
    "camel": "Camel",
    "beige": "Bej",
    "cream": "Krem",
    "off-white": "Kırık Beyaz",
    "ivory": "Kırık Beyaz",
    "yellow": "Sarı",
    "mustard": "Hardal",
    "orange": "Turuncu",
    "pink": "Pembe",
    "hot pink": "Fuşya",
    "fuchsia": "Fuşya",
    "rose": "Gül Kurusu",
    "purple": "Mor",
    "lavender": "Lavanta",
    "lilac": "Leylak",
    "violet": "Mor",
    "silver": "Gümüş",
    "gold": "Altın",
    "metallic": "Metalik",
    "denim": "Denim",
    "indigo": "İndigo",
    "coral": "Mercan",
    "teal": "Petrol",
    "turquoise": "Turkuaz",
    "charcoal": "Antrasit",
    "multicolor": "Çok Renkli",
    "multi": "Çok Renkli",
    "striped": "Çizgili",
    "plaid": "Ekose",
}

ITEM_TRANSLATIONS: Dict[str, str] = {
    # Tops
    "t-shirt": "Tişört",
    "tshirt": "Tişört",
    "shirt": "Gömlek",
    "blouse": "Bluz",
    "top": "Üst",
    "sweater": "Kazak",
    "pullover": "Kazak",
    "knitwear": "Triko",
    "hoodie": "Kapüşonlu Sweatshirt",
    "sweatshirt": "Sweatshirt",
    "cardigan": "Hırka",
    "vest": "Yelek",
    # Outerwear
    "jacket": "Ceket",
    "blazer": "Blazer Ceket",
    "coat": "Mont",
    "trench coat": "Trençkot",
    "parka": "Parka",
    "windbreaker": "Yağmurluk",
    "leather jacket": "Deri Ceket",
    # Bottoms
    "pants": "Pantolon",
    "trousers": "Pantolon",
    "jeans": "Jean",
    "denim": "Jean",
    "shorts": "Şort",
    "skirt": "Etek",
    "mini skirt": "Mini Etek",
    # Full body
    "dress": "Elbise",
    "maxi dress": "Maksi Elbise",
    "mini dress": "Mini Elbise",
    "jumpsuit": "Tulum",
    "overalls": "Salopet",
    # Footwear
    "shoes": "Ayakkabı",
    "sneakers": "Spor Ayakkabı",
    "boots": "Bot",
    "heels": "Topuklu Ayakkabı",
    "sandals": "Sandalet",
    "loafers": "Loafer",
    "oxfords": "Oxford Ayakkabı",
    # Accessories
    "bag": "Çanta",
    "handbag": "El Çantası",
    "backpack": "Sırt Çantası",
    "belt": "Kemer",
    "watch": "Saat",
    "scarf": "Atkı",
    "hat": "Şapka",
    "cap": "Kep",
    "sunglasses": "Güneş Gözlüğü",
}

PATTERN_TRANSLATIONS: Dict[str, str] = {
    "solid": "",  # Don't add "düz" - just leave blank, it's implicit
    "striped": "Çizgili",
    "plaid": "Ekose",
    "checked": "Ekose",
    "floral": "Çiçekli",
    "graphic": "Baskılı",
    "polka-dot": "Puanlı",
    "animal print": "Hayvan Desenli",
    "geometric": "Geometrik",
    "abstract": "Desenli",
    "camo": "Kamuflaj",
    "camouflage": "Kamuflaj",
}

GENDER_TRANSLATIONS: Dict[str, str] = {
    "male": "Erkek",
    "female": "Kadın",
    "unisex": "",
}

# ---------------------------------------------------------------------------
# Fashion Analysis Prompt
# ---------------------------------------------------------------------------

FASHION_ANALYSIS_PROMPT = """You are a professional fashion analyst. Analyze this image carefully.

Return ONLY a valid JSON object (no markdown, no explanation) with this exact structure:
{
  "gender": "male or female or unisex",
  "items": [
    {
      "type": "one of: t-shirt, shirt, blouse, sweater, hoodie, sweatshirt, cardigan, vest, jacket, blazer, coat, trench coat, parka, leather jacket, pants, jeans, shorts, skirt, dress, jumpsuit, shoes, sneakers, boots, heels, sandals, loafers, bag, belt, watch, scarf, hat, cap, sunglasses",
      "color": "precise color (e.g. navy blue, cream, burgundy, olive green, charcoal)",
      "pattern": "one of: solid, striped, plaid, floral, graphic, polka-dot, geometric, animal print, camo",
      "material": "one of: denim, cotton, wool, leather, silk, knit, polyester, linen, synthetic, unknown",
      "fit": "one of: slim, regular, oversized, wide-leg, cropped, fitted, relaxed, unknown",
      "description": "one precise sentence about this item only"
    }
  ],
  "overall_style": "one of: casual, smart-casual, formal, sporty, streetwear, bohemian, minimalist, elegant",
  "occasion": "one of: everyday, work, evening, sport, beach, formal, party"
}

Rules:
- Only include items CLEARLY VISIBLE in the image
- Be very precise about colors (say "navy blue" not just "blue")
- For gender: use visible cues (clothing cut, styling) - default to unisex if unclear
- List items from most to least prominent
- Maximum 5 items"""


# ---------------------------------------------------------------------------
# VLMService
# ---------------------------------------------------------------------------

class VLMService:
    """
    Fashion image analysis using Qwen2-VL-7B-Instruct via HF Inference API.

    Single structured prompt → JSON output → Turkish search queries.
    No local model loading, no regex heuristics.
    """

    def __init__(self, model_name: str = VLM_MODEL_NAME):
        self.model_name = model_name
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Initialize HuggingFace InferenceClient."""
        try:
            from huggingface_hub import InferenceClient
            if not HF_API_TOKEN:
                logger.warning("HF_API_TOKEN not set — VLM calls will fail or be rate-limited")
            self.client = InferenceClient(token=HF_API_TOKEN or None)
            logger.info(f"✅ InferenceClient initialized for model: {self.model_name}")
        except ImportError:
            logger.error("huggingface_hub not installed. Run: pip install huggingface-hub>=0.23")
            self.client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_fashion_image(self, image: Image.Image) -> Dict:
        """
        Analyze a fashion image and return structured attribute data.

        Args:
            image: PIL Image (already preprocessed)

        Returns:
            Dict with keys: items, gender, overall_style, occasion
        """
        if self.client is None:
            return self._empty_result("HF InferenceClient not available")

        image_b64_url = self._image_to_data_uri(image)

        # Try primary model, then fallback
        raw_text = self._call_vlm(image_b64_url, attempt=1)
        if raw_text is None:
            logger.warning("Primary VLM call failed, retrying after delay...")
            time.sleep(10)
            raw_text = self._call_vlm(image_b64_url, attempt=2)

        if raw_text is None:
            return self._empty_result("VLM API unavailable after retries. Check HF_API_TOKEN.")

        fashion_data = self._parse_vlm_json(raw_text)
        logger.info(f"✅ Detected {len(fashion_data.get('items', []))} items: "
                    f"{[i.get('type') for i in fashion_data.get('items', [])]}")
        return fashion_data

    def get_search_queries(self, fashion_data: Dict) -> List[str]:
        """
        Convert structured fashion data to Turkish Trendyol search queries.

        Args:
            fashion_data: Output from analyze_fashion_image()

        Returns:
            List of Turkish search query strings (max 5)
        """
        queries = []
        gender_tr = GENDER_TRANSLATIONS.get(
            fashion_data.get("gender", "unisex"), ""
        )

        for item in fashion_data.get("items", [])[:5]:
            item_type = item.get("type", "").lower().strip()
            color = item.get("color", "").lower().strip()
            pattern = item.get("pattern", "solid").lower().strip()

            item_tr = ITEM_TRANSLATIONS.get(item_type, "")
            if not item_tr:
                # Try partial match
                for key, val in ITEM_TRANSLATIONS.items():
                    if key in item_type or item_type in key:
                        item_tr = val
                        break
            if not item_tr:
                continue  # Skip unknown item types

            color_tr = COLOR_TRANSLATIONS.get(color, "")
            if not color_tr:
                # Try partial match for compound colors like "dark navy blue"
                for key, val in COLOR_TRANSLATIONS.items():
                    if key in color:
                        color_tr = val
                        break

            pattern_tr = PATTERN_TRANSLATIONS.get(pattern, "")

            parts = [p for p in [gender_tr, color_tr, pattern_tr, item_tr] if p]
            query = " ".join(parts)
            if query and query not in queries:
                queries.append(query)

        logger.info(f"Generated search queries: {queries}")
        return queries

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 data URI for API submission."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffered = BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        b64 = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def _call_vlm(self, image_data_uri: str, attempt: int = 1) -> Optional[str]:
        """
        Send image + prompt to Qwen2-VL-7B-Instruct via chat_completion.

        Returns raw text response or None on failure.
        """
        try:
            logger.info(f"Calling VLM (attempt {attempt}): {self.model_name}")
            response = self.client.chat_completion(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_uri}
                            },
                            {
                                "type": "text",
                                "text": FASHION_ANALYSIS_PROMPT
                            }
                        ]
                    }
                ],
                max_tokens=1024,
                temperature=0.1,   # Low temp = factual, consistent output
            )
            text = response.choices[0].message.content
            logger.debug(f"VLM raw response: {text[:300]}...")
            return text

        except Exception as e:
            err = str(e)
            if "503" in err or "loading" in err.lower():
                logger.warning(f"Model loading (503) on attempt {attempt}")
            elif "429" in err or "rate" in err.lower():
                logger.warning(f"Rate limited (429) on attempt {attempt}")
            elif "401" in err or "unauthorized" in err.lower():
                logger.error("Invalid HF_API_TOKEN — check your token")
            else:
                logger.error(f"VLM call failed: {e}")
            return None

    def _parse_vlm_json(self, raw_text: str) -> Dict:
        """
        Extract and parse JSON from the VLM response.

        Handles:
        - Clean JSON responses
        - Responses wrapped in ```json ... ``` fences
        - Responses with extra text before/after JSON
        """
        if not raw_text:
            return self._empty_result("Empty VLM response")

        # Strip markdown code fences if present
        text = raw_text.strip()
        fence_match = re.search(r'```(?:json)?\s*([\s\S]+?)\s*```', text)
        if fence_match:
            text = fence_match.group(1).strip()

        # If no fences, try to find JSON block by braces
        if not text.startswith('{'):
            brace_match = re.search(r'\{[\s\S]+\}', text)
            if brace_match:
                text = brace_match.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse VLM JSON: {e}\nRaw: {text[:500]}")
            # Try to recover partial JSON
            data = self._recover_partial_json(text)
            if not data:
                return self._empty_result(f"Invalid JSON from VLM: {str(e)[:100]}")

        return self._normalize_fashion_data(data)

    def _normalize_fashion_data(self, data: Dict) -> Dict:
        """Ensure all required fields exist with sensible defaults."""
        result = {
            "gender": data.get("gender", "unisex"),
            "items": [],
            "overall_style": data.get("overall_style", "casual"),
            "occasion": data.get("occasion", "everyday"),
        }

        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            normalized = {
                "type": item.get("type", "shirt").lower().strip(),
                "color": item.get("color", "unknown").lower().strip(),
                "pattern": item.get("pattern", "solid").lower().strip(),
                "material": item.get("material", "unknown").lower().strip(),
                "fit": item.get("fit", "unknown").lower().strip(),
                "description": item.get("description", ""),
                # Compatibility fields for existing UI code
                "style": result["overall_style"],
            }
            result["items"].append(normalized)

        return result

    def _recover_partial_json(self, text: str) -> Optional[Dict]:
        """Attempt to recover usable data from malformed JSON."""
        try:
            # Try fixing common issues: trailing comma, missing closing brace
            fixed = re.sub(r',\s*}', '}', text)
            fixed = re.sub(r',\s*]', ']', fixed)
            if not fixed.rstrip().endswith('}'):
                fixed = fixed.rstrip() + '}'
            return json.loads(fixed)
        except Exception:
            return None

    @staticmethod
    def _empty_result(error_msg: str = "") -> Dict:
        """Return a safely-typed empty result dict."""
        if error_msg:
            logger.warning(f"Returning empty result: {error_msg}")
        return {
            "gender": "unisex",
            "items": [],
            "overall_style": "unknown",
            "occasion": "unknown",
            "error": error_msg,
        }

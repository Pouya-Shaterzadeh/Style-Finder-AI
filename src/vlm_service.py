"""
Vision Language Model Service — Style Finder AI

Uses GPT OSS 120B on GroqCloud for fashion image analysis:
- Free tier: ~14,400 req/day, no credit card required
- Fast inference via Groq's LPU hardware
- Single structured prompt → {gender, items[], overall_style, occasion}
- Visual similarity: patrickjohncyh/fashion-clip (separate, in image_processor.py)
"""

import re
import json
import time
import logging
import os
import sys
from io import BytesIO
from typing import Dict, List, Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(log_dir, exist_ok=True)

_fh = logging.FileHandler(os.path.join(log_dir, "vlm_service.log"))
_fh.setLevel(logging.DEBUG)
_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s - %(levelname)s - [VLM] %(message)s")
_fh.setFormatter(_fmt)
_ch.setFormatter(_fmt)
if not logger.handlers:
    logger.addHandler(_fh)
    logger.addHandler(_ch)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import GROQ_API_KEY, GROQ_MODEL

# ---------------------------------------------------------------------------
# Turkish Translation Tables
# ---------------------------------------------------------------------------

COLOR_TRANSLATIONS: Dict[str, str] = {
    "black": "Siyah", "white": "Beyaz", "gray": "Gri", "grey": "Gri",
    "navy": "Lacivert", "navy blue": "Lacivert", "blue": "Mavi",
    "light blue": "Açık Mavi", "dark blue": "Koyu Mavi", "royal blue": "Saks Mavi",
    "red": "Kırmızı", "dark red": "Koyu Kırmızı", "burgundy": "Bordo",
    "maroon": "Bordo", "wine": "Bordo", "green": "Yeşil",
    "dark green": "Koyu Yeşil", "olive": "Haki", "olive green": "Haki",
    "khaki": "Haki", "mint": "Mint", "mint green": "Mint",
    "brown": "Kahverengi", "dark brown": "Koyu Kahverengi",
    "tan": "Camel", "camel": "Camel", "beige": "Bej", "cream": "Krem",
    "off-white": "Kırık Beyaz", "ivory": "Kırık Beyaz",
    "yellow": "Sarı", "mustard": "Hardal", "orange": "Turuncu",
    "pink": "Pembe", "hot pink": "Fuşya", "fuchsia": "Fuşya",
    "rose": "Gül Kurusu", "purple": "Mor", "lavender": "Lavanta",
    "lilac": "Leylak", "violet": "Mor", "silver": "Gümüş", "gold": "Altın",
    "metallic": "Metalik", "denim": "Denim", "indigo": "İndigo",
    "coral": "Mercan", "teal": "Petrol", "turquoise": "Turkuaz",
    "charcoal": "Antrasit", "multicolor": "Çok Renkli", "multi": "Çok Renkli",
    "striped": "Çizgili", "plaid": "Ekose",
}

ITEM_TRANSLATIONS: Dict[str, str] = {
    # Tops
    "t-shirt": "Tişört", "tshirt": "Tişört", "shirt": "Gömlek",
    "blouse": "Bluz", "top": "Üst", "sweater": "Kazak",
    "pullover": "Kazak", "knitwear": "Triko", "hoodie": "Kapüşonlu Sweatshirt",
    "sweatshirt": "Sweatshirt", "cardigan": "Hırka", "vest": "Yelek",
    # Outerwear
    "jacket": "Ceket", "blazer": "Blazer Ceket", "coat": "Mont",
    "trench coat": "Trençkot", "parka": "Parka", "windbreaker": "Yağmurluk",
    "leather jacket": "Deri Ceket",
    # Bottoms
    "pants": "Pantolon", "trousers": "Pantolon", "jeans": "Jean",
    "denim": "Jean", "shorts": "Şort", "skirt": "Etek",
    "mini skirt": "Mini Etek",
    # Full body
    "dress": "Elbise", "maxi dress": "Maksi Elbise",
    "mini dress": "Mini Elbise", "jumpsuit": "Tulum", "overalls": "Salopet",
    # Footwear
    "shoes": "Ayakkabı", "sneakers": "Spor Ayakkabı", "boots": "Bot",
    "heels": "Topuklu Ayakkabı", "sandals": "Sandalet",
    "loafers": "Loafer", "oxfords": "Oxford Ayakkabı",
    # Accessories
    "bag": "Çanta", "handbag": "El Çantası", "backpack": "Sırt Çantası",
    "belt": "Kemer", "watch": "Saat", "scarf": "Atkı",
    "hat": "Şapka", "cap": "Kep", "sunglasses": "Güneş Gözlüğü",
}

PATTERN_TRANSLATIONS: Dict[str, str] = {
    "solid": "",
    "striped": "Çizgili", "plaid": "Ekose", "checked": "Ekose",
    "floral": "Çiçekli", "graphic": "Baskılı", "polka-dot": "Puanlı",
    "animal print": "Hayvan Desenli", "geometric": "Geometrik",
    "abstract": "Desenli", "camo": "Kamuflaj", "camouflage": "Kamuflaj",
}

GENDER_TRANSLATIONS: Dict[str, str] = {
    "male": "Erkek", "female": "Kadın", "unisex": "",
}

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

FASHION_ANALYSIS_PROMPT = """You are a senior fashion editor and personal stylist. Study this image carefully before writing anything.

Return ONLY a valid JSON object with this exact structure:
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
  "occasion": "one of: everyday, work, evening, sport, beach, formal, party",
  "stylist_notes": [
    "COLOR PALETTE — evaluate the specific color combination you see: do the exact colors complement each other? Name the colors and give a precise verdict with one concrete action (e.g. 'The olive cargo pants and cream ribbed top form a grounded, earthy pairing — swap white sneakers for tan leather boots to stay in the warm palette.').",
    "FIT & PROPORTION — comment on the silhouette the visible items create together: are the proportions balanced? Call out any specific imbalance (oversized vs slim, cropped vs high-waisted) and state clearly whether it works or exactly how to fix it.",
    "FINISHING TOUCH — identify the single most impactful item missing from this outfit and name it precisely with a color (e.g. 'A thin cognac leather belt would anchor the high-waisted trousers and add definition to the waist that is currently lost under the relaxed blouse.')."
  ]
}

Rules:
- Only include items CLEARLY VISIBLE in the image
- Be very precise about colors (say "navy blue" not just "blue")
- For gender: use visible cues (clothing cut, styling) — default to unisex if unclear
- List items from most to least prominent
- Maximum 5 items
- stylist_notes: write exactly 3 notes following the COLOR PALETTE / FIT & PROPORTION / FINISHING TOUCH structure; each note must be 1-2 sentences; ALWAYS reference the specific colors and items you actually observe; NO generic advice"""


# ---------------------------------------------------------------------------
# VLMService
# ---------------------------------------------------------------------------

class VLMService:
    """
    Fashion image analysis using GPT OSS 120B on GroqCloud.

    Groq runs inference on its LPU hardware — no local GPU needed.
    Free tier: ~14,400 req/day, 30 RPM.
    Get a free API key at: https://console.groq.com
    """

    def __init__(self):
        self.client = None
        self._setup_groq()

    def _setup_groq(self):
        """Initialize the Groq client."""
        if not GROQ_API_KEY:
            logger.error("GROQ_API_KEY is not set. Add it to .env or HF Spaces Secrets.")
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=GROQ_API_KEY)
            logger.info(f"✅ Groq client initialized: {GROQ_MODEL}")
        except ImportError:
            logger.error("groq not installed. Run: pip install groq")
        except Exception as e:
            logger.error(f"Failed to initialize Groq: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_fashion_image(self, image: Image.Image) -> Dict:
        """
        Analyze a fashion image with GPT OSS 120B on GroqCloud.
        Returns structured dict: {gender, items[], overall_style, occasion, stylist_notes[]}
        """
        if self.client is None:
            return self._empty_result(
                "Groq API not configured. Set GROQ_API_KEY in environment."
            )

        image_bytes = self._pil_to_jpeg_bytes(image)

        # Attempt 1
        raw = self._call_groq(image_bytes, attempt=1)

        # On rate-limit (429) retry after 60 s
        if raw is None:
            logger.warning("Groq call failed — retrying in 60 s...")
            time.sleep(60)
            raw = self._call_groq(image_bytes, attempt=2)

        if raw is None:
            return self._empty_result(
                "Groq API unavailable after retries. "
                "Check your GROQ_API_KEY or try again shortly."
            )

        fashion_data = self._parse_vlm_json(raw)
        logger.info(
            f"✅ Detected {len(fashion_data.get('items', []))} items: "
            f"{[i.get('type') for i in fashion_data.get('items', [])]}"
        )
        return fashion_data

    def get_search_queries(self, fashion_data: Dict) -> List[str]:
        """
        Convert structured fashion data to Turkish Trendyol search queries.
        e.g. female + navy blue + slim + jeans → "Kadın Lacivert Slim Jean"
        """
        queries = []
        gender_tr = GENDER_TRANSLATIONS.get(
            fashion_data.get("gender", "unisex"), ""
        )

        for item in fashion_data.get("items", [])[:5]:
            item_type = item.get("type", "").lower().strip()
            color     = item.get("color", "").lower().strip()
            pattern   = item.get("pattern", "solid").lower().strip()

            item_tr = ITEM_TRANSLATIONS.get(item_type, "")
            if not item_tr:
                for key, val in ITEM_TRANSLATIONS.items():
                    if key in item_type or item_type in key:
                        item_tr = val
                        break
            if not item_tr:
                continue

            color_tr = COLOR_TRANSLATIONS.get(color, "")
            if not color_tr:
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

    def _pil_to_jpeg_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to JPEG bytes."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _call_groq(self, image_bytes: bytes, attempt: int = 1) -> Optional[str]:
        """
        Call GPT OSS 120B via the Groq API with the image and fashion prompt.
        Image is sent as a base64-encoded data URI.
        Returns raw response text or None on failure.
        """
        import base64
        try:
            logger.info(f"Calling Groq (attempt {attempt}): {GROQ_MODEL}")

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                },
                            },
                            {
                                "type": "text",
                                "text": FASHION_ANALYSIS_PROMPT,
                            },
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=1024,
            )

            text = response.choices[0].message.content
            logger.debug(f"Groq raw response: {text[:300]}...")
            return text

        except Exception as e:
            err = repr(e)
            logger.error(
                f"Groq call failed (attempt {attempt}): "
                f"type={type(e).__name__} | {err[:300]}"
            )
            return None

    def _parse_vlm_json(self, raw_text: str) -> Dict:
        """
        Parse JSON from Groq response.
        Handles markdown fences and extracts the first JSON object found.
        """
        if not raw_text:
            return self._empty_result("Empty response from Groq")

        text = raw_text.strip()

        # Strip markdown fences if present (fallback)
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fence:
            text = fence.group(1).strip()

        # If still not a JSON object, try to extract one
        if not text.startswith("{"):
            brace = re.search(r"\{[\s\S]+\}", text)
            if brace:
                text = brace.group(0)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw: {text[:500]}")
            data = self._recover_partial_json(text)
            if not data:
                return self._empty_result(f"Invalid JSON from Groq: {str(e)[:100]}")

        return self._normalize_fashion_data(data)

    def _normalize_fashion_data(self, data: Dict) -> Dict:
        """Ensure all required fields exist with sensible defaults."""
        result = {
            "gender":        data.get("gender", "unisex"),
            "items":         [],
            "overall_style": data.get("overall_style", "casual"),
            "occasion":      data.get("occasion", "everyday"),
            "stylist_notes": [n for n in data.get("stylist_notes", []) if isinstance(n, str) and n.strip()][:3],
        }
        for item in data.get("items", []):
            if not isinstance(item, dict):
                continue
            result["items"].append({
                "type":        item.get("type", "shirt").lower().strip(),
                "color":       item.get("color", "unknown").lower().strip(),
                "pattern":     item.get("pattern", "solid").lower().strip(),
                "material":    item.get("material", "unknown").lower().strip(),
                "fit":         item.get("fit", "unknown").lower().strip(),
                "description": item.get("description", ""),
                "style":       result["overall_style"],
            })
        return result

    def _recover_partial_json(self, text: str) -> Optional[Dict]:
        """Best-effort recovery for slightly malformed JSON."""
        try:
            fixed = re.sub(r",\s*}", "}", text)
            fixed = re.sub(r",\s*]", "]", fixed)
            if not fixed.rstrip().endswith("}"):
                fixed = fixed.rstrip() + "}"
            return json.loads(fixed)
        except Exception:
            return None

    @staticmethod
    def _empty_result(error_msg: str = "") -> Dict:
        if error_msg:
            logger.warning(f"Returning empty result: {error_msg}")
        return {
            "gender":        "unisex",
            "items":         [],
            "overall_style": "unknown",
            "occasion":      "unknown",
            "stylist_notes": [],
            "error":         error_msg,
        }

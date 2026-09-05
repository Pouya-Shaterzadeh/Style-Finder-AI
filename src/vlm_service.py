"""
Vision Language Model Service — Style Finder AI

Uses Gemini 2.5 Flash on Google AI Studio for fashion image analysis:
- Free tier: 1,500 req/day, no credit card required
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
from typing import Dict, List, Tuple, Optional

from PIL import Image

# Versioned prompts and tracing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prompts import FASHION_ANALYSIS, VERSION as PROMPT_VERSION
from tracing import log_llm_call

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
from config.config import GOOGLE_API_KEY, GEMINI_MODEL

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
    # Trendyol-specific neutrals
    "taupe": "Taş", "greige": "Taş", "stone": "Taş", "mushroom": "Taş",
    "ecru": "Ekru", "oatmeal": "Bej", "sand": "Bej",
    # Compound colors (for corrected VLM output)
    "blue cream": "Mavi Krem", "navy white": "Lacivert Beyaz",
    "blue white": "Mavi Beyaz", "navy cream": "Lacivert Krem",
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

MATERIAL_TRANSLATIONS: Dict[str, str] = {
    "knit": "Triko", "knitted": "Triko", "ribbed knit": "Fitilli Triko",
    "wool": "Yün", "cotton": "Pamuklu", "denim": "Kot", "denim cotton": "Kot",
    "leather": "Deri", "suede": "Süet", "silk": "İpek", "satin": "Saten",
    "polyester": "Polyester", "linen": "Keten", "linen blend": "Keten",
    "synthetic": "", "unknown": "", "": "",
    # compound / texture hints
    "ribbed": "Fitilli", "rib knit": "Fitilli Triko", "cable knit": "Saç Örgü Triko",
    "fleece": "Polar", "cashmere": "Kaşmir", "velvet": "Kadife",
}

FIT_TRANSLATIONS: Dict[str, str] = {
    "slim": "Slim", "skinny": "Skinny", "fitted": "Dar", "tailored": "Dar",
    "regular": "", "relaxed": "Bol", "loose": "Bol", "oversized": "Oversize",
    "baggy": "Bol", "wide-leg": "Geniş Paça", "wide": "Geniş Paça",
    "wide leg": "Geniş Paça", "flare": "Klapa", "flared": "Klapa",
    "straight": "Düz", "straight-leg": "Düz Paça", "straight leg": "Düz Paça",
    "cropped": "Kısa", "cropped": "Kısa", "high-waisted": "Yüksek Bel",
    "high waist": "Yüksek Bel", "mid-rise": "Orta Bel", "low-rise": "Düşük Bel",
    "bootcut": "Bootcut", "cargo": "Kargo", "sweatpants": "Eşofman",
    "jogger": "Jogger", "unknown": "", "": "",
}

# Neckline / collar and sleeve — extracted from VLM description for precision
YAKA_KEYWORDS: List[Tuple[str, str]] = [
    ("square neckline", "Kare Yaka"), ("square neck", "Kare Yaka"), ("square-neck", "Kare Yaka"),
    ("polo", "Polo Yaka"), ("turtleneck", "Balıkçı Yaka"), ("high neck", "Balıkçı Yaka"),
    ("mock neck", "Balıkçı Yaka"), ("boat neck", "Kayık Yaka"), ("off-shoulder", "Omzu Açık"),
    ("off shoulder", "Omzu Açık"), ("off the shoulder", "Omzu Açık"),
    ("v-neck", "V Yaka"), ("v neck", "V Yaka"), ("v-neckline", "V Yaka"),
    ("crew neck", "Bisiklet Yaka"), ("round neck", "Bisiklet Yaka"), ("scoop neck", "Bisiklet Yaka"),
    ("halter", "Halter Yaka"), ("one-shoulder", "Tek Omuz"), ("collared", "Gömlek Yaka"),
]

SLEEVE_KEYWORDS: List[Tuple[str, str]] = [
    ("long-sleeve", "Uzun Kollu"), ("long sleeve", "Uzun Kollu"), ("long sleeves", "Uzun Kollu"),
    ("short-sleeve", "Kısa Kollu"), ("short sleeve", "Kısa Kollu"), ("short sleeves", "Kısa Kollu"),
    ("sleeveless", "Kolsuz"), ("tank", "Kolsuz"), ("three-quarter", "Yarım Kollu"),
    ("3/4", "Yarım Kollu"), ("cap sleeve", "Kısa Kollu"), ("elbow", "Dirsek Kollu"),
]

# ---------------------------------------------------------------------------
# Color Hallucination Corrections — Post-process VLM output to fix common errors
# ---------------------------------------------------------------------------

# Map of hallucinated colors → likely actual colors based on common VLM failures
# Keys match VLM output (English) — corrections applied BEFORE Turkish translation
COLOR_HALLUCINATION_CORRECTIONS: Dict[str, List[Tuple[str, str, str]]] = {
    # Format: item_type -> [(hallucinated_color, pattern_context, corrected_color), ...]
    "shirt": [
        ("black", "plaid", "blue cream"),        # blue/cream plaid → NOT black
        ("black", "striped", "navy white"),      # navy/white stripes → NOT black
        ("black", "solid", "navy"),              # dark navy solid → navy
        ("dark blue", "plaid", "navy cream"),    # dark blue plaid → navy/cream
    ],
    "pants": [
        ("black", "beige", "navy"),              # beige pants with dark top → navy
        ("black", "cream", "charcoal"),          # cream combo → charcoal
    ],
    "t-shirt": [
        ("black", "", "navy"),                   # default dark → navy
    ],
}

def correct_color_hallucinations(items: List[Dict]) -> List[Dict]:
    """
    Post-process VLM output to fix common color hallucinations.
    Especially: 'siyah' (black) hallucinated for blue/cream plaid shirts.
    """
    corrected = []
    for item in items:
        item_type = item.get("type", "").lower().strip()
        color = item.get("color", "").lower().strip()
        pattern = item.get("pattern", "").lower().strip()
        
        if item_type in COLOR_HALLUCINATION_CORRECTIONS:
            for hallucinated, pattern_ctx, corrected_color in COLOR_HALLUCINATION_CORRECTIONS[item_type]:
                # Match if hallucinated color appears anywhere in VLM's color string
                # AND pattern context matches (or empty = any)
                if hallucinated in color and (not pattern_ctx or pattern_ctx in pattern):
                    logger.warning(f"Color hallucination corrected: {item_type} '{color}' + pattern '{pattern}' → '{corrected_color}'")
                    item["color"] = corrected_color
                    break  # Only apply first matching correction
        
        corrected.append(item)
    return corrected


# ---------------------------------------------------------------------------
# Prompt (versioned — see prompts.py)
# ---------------------------------------------------------------------------

FASHION_ANALYSIS_PROMPT = FASHION_ANALYSIS["prompt"]


# ---------------------------------------------------------------------------
# VLMService
# ---------------------------------------------------------------------------

class VLMService:
    """
    Fashion image analysis using Gemini 2.5 Flash on Google AI Studio.

    Free tier: 1,500 req/day, no credit card required.
    Get a free API key at: https://aistudio.google.com/apikey
    """

    def __init__(self):
        self.client = None
        self._setup_client()

    def _setup_client(self):
        """Initialize the OpenAI-compatible client for Google AI Studio."""
        if not GOOGLE_API_KEY:
            logger.error("GOOGLE_API_KEY is not set. Add it to .env or HF Spaces Secrets.")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=GOOGLE_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            logger.info(f"✅ Google AI Studio client initialized: {GEMINI_MODEL}")
        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
        except Exception as e:
            logger.error(f"Failed to initialize Google AI Studio client: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_fashion_image(self, image: Image.Image) -> Dict:
        """
        Analyze a fashion image with Gemini 2.5 Flash on Google AI Studio.
        Returns structured dict: {gender, items[], overall_style, occasion, stylist_notes[]}
        """
        if self.client is None:
            return self._empty_result(
                "Google AI Studio not configured. Set GOOGLE_API_KEY in environment."
            )

        image_bytes = self._pil_to_jpeg_bytes(image)

        # Attempt 1
        raw = self._call_api(image_bytes, attempt=1)

        # On rate-limit (429) retry after 60 s
        if raw is None:
            logger.warning("API call failed — retrying in 60 s...")
            time.sleep(60)
            raw = self._call_api(image_bytes, attempt=2)

        if raw is None:
            return self._empty_result(
                "API unavailable after retries. "
                "Check your GOOGLE_API_KEY or try again shortly."
            )

        fashion_data = self._parse_vlm_json(raw)
        
        # DEBUG: Log raw VLM output before correction
        if fashion_data.get("items"):
            for i, item in enumerate(fashion_data["items"]):
                logger.info(f"VLM item {i}: type={item.get('type')} color={item.get('color')} pattern={item.get('pattern')}")
        
        # Apply color hallucination corrections
        if fashion_data.get("items"):
            fashion_data["items"] = correct_color_hallucinations(fashion_data["items"])
        
        # DEBUG: Log after correction
        if fashion_data.get("items"):
            for i, item in enumerate(fashion_data["items"]):
                logger.info(f"POST-CORRECTION item {i}: type={item.get('type')} color={item.get('color')} pattern={item.get('pattern')}")

        logger.info(
            f"✅ Detected {len(fashion_data.get('items', []))} items: "
            f"{[i.get('type') for i in fashion_data.get('items', [])]}"
        )
        return fashion_data

    def get_search_queries(self, fashion_data: Dict) -> List[str]:
        """
        Convert structured fashion data to Turkish Trendyol search queries.
        Includes texture/material so knit vs satin vs cotton don't collapse.
        e.g. male burgundy knit polo → "Erkek Bordo Fitilli Triko Polo Yaka Gömlek"
        """
        queries = []
        gender_tr = GENDER_TRANSLATIONS.get(
            fashion_data.get("gender", "unisex"), ""
        )

        for item in fashion_data.get("items", [])[:5]:
            item_type   = item.get("type", "").lower().strip()
            color       = item.get("color", "").lower().strip()
            pattern     = item.get("pattern", "solid").lower().strip()
            fit         = item.get("fit", "").lower().strip()
            material    = item.get("material", "").lower().strip()
            description = item.get("description", "").lower()

            # ---- Item translation with polo detection ----
            # VLM often returns "shirt" for polo knits; use description to refine.
            is_polo = "polo" in description or "polo" in item_type
            base_item = item_type
            if is_polo and item_type in ("shirt", "t-shirt", "tshirt"):
                item_tr = "Polo Yaka Gömlek" if item_type == "shirt" else "Polo Yaka Tişört"
            else:
                item_tr = ITEM_TRANSLATIONS.get(item_type, "")
                if not item_tr:
                    for key, val in ITEM_TRANSLATIONS.items():
                        if key in item_type or item_type in key:
                            item_tr = val
                            break
            if not item_tr:
                continue

            # Color — exact match first, then partial substring match
            color_tr = COLOR_TRANSLATIONS.get(color, "")
            if not color_tr:
                for key, val in COLOR_TRANSLATIONS.items():
                    if key in color:
                        color_tr = val
                        break
            if not color_tr:
                color_tr = color.capitalize()

            pattern_tr = PATTERN_TRANSLATIONS.get(pattern, "")
            if not pattern_tr:
                for key, val in PATTERN_TRANSLATIONS.items():
                    if key in pattern:
                        pattern_tr = val
                        break

            # Material / texture — critical for knit vs satin distinction
            material_tr = MATERIAL_TRANSLATIONS.get(material, "")
            if not material_tr:
                for key, val in MATERIAL_TRANSLATIONS.items():
                    if key and key in material:
                        material_tr = val
                        break
            # Enhance with description texture hints when material is generic
            if "ribbed" in description or "rib knit" in description:
                if material_tr and "Fitilli" not in material_tr:
                    material_tr = f"Fitilli {material_tr}".strip()
                elif not material_tr:
                    material_tr = "Fitilli Triko"
            if "satin" in description:
                material_tr = "Saten"

            # Neckline / yaka and sleeve — parse from description (high precision)
            yaka_tr = ""
            for kw, tr in YAKA_KEYWORDS:
                if kw in description:
                    yaka_tr = tr
                    break
            # Avoid duplicate Polo Yaka already in item name
            if yaka_tr and yaka_tr in item_tr:
                yaka_tr = ""

            kol_tr = ""
            for kw, tr in SLEEVE_KEYWORDS:
                if kw in description:
                    kol_tr = tr
                    break

            # Fit / model — deprioritize generic Dar when yaka is more distinctive for tops
            fit_tr = FIT_TRANSLATIONS.get(fit, "")
            if not fit_tr:
                for key, val in FIT_TRANSLATIONS.items():
                    if key in fit:
                        fit_tr = val
                        break
            # For upper-body knit tops, neckline matters more than tightness
            if yaka_tr and fit_tr == "Dar" and item_type in ("sweater", "shirt", "blouse", "t-shirt", "tshirt", "top", "cardigan", "knitwear", "pullover"):
                fit_tr = ""

            # ----- Footwear simplification — avoid Taupe/Deri/Çizgili hallucinations
            _FOOTWEAR = {"shoes", "sneakers", "boots", "heels", "sandals", "loafers", "oxfords"}
            if item_type in _FOOTWEAR:
                # striped/plaid solid hallucinations are common for laces/soles — keep only distinctive patterns
                if pattern in ("striped", "plaid", "checked", "solid"):
                    pattern_tr = ""
                # Sneakers are mostly synthetic/canvas — Deri is often hallucinated; drop to avoid over-filtering
                if item_type == "sneakers":
                    material_tr = ""
                # Taupe sole vs white upper — prioritize upper; if shoe is white-dominant, taupe is likely sole hallucination
                if color == "taupe" and ("white" in description or "cream" in description or "bej" in description.lower()):
                    color_tr = "Beyaz"

            # Avoid duplicate material in item name (e.g. Deri + Deri Ceket)
            if material_tr and material_tr in item_tr:
                material_tr = ""

            parts = [p for p in [gender_tr, color_tr, material_tr, yaka_tr, kol_tr, pattern_tr, fit_tr, item_tr] if p]
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

    def _call_api(self, image_bytes: bytes, attempt: int = 1) -> Optional[str]:
        """
        Call Gemini 2.5 Flash via Google AI Studio with the image and fashion prompt.
        Image is sent as a base64-encoded data URI.
        Returns raw response text or None on failure.
        """
        import base64
        try:
            logger.info(f"Calling Google AI Studio (attempt {attempt}): {GEMINI_MODEL}")

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            start = time.monotonic()
            response = self.client.chat.completions.create(
                model=GEMINI_MODEL,
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
                max_tokens=4096,
            )

            latency_ms = int((time.monotonic() - start) * 1000)
            text = response.choices[0].message.content
            logger.debug(f"Gemini raw response: {text[:300]}...")

            # Log to LangSmith
            log_llm_call(
                prompt_name="fashion_analysis",
                prompt_version=PROMPT_VERSION,
                model=GEMINI_MODEL,
                image_bytes=image_bytes,
                response_text=text,
                latency_ms=latency_ms,
                temperature=0.1,
                max_tokens=4096,
            )

            return text

        except Exception as e:
            err = repr(e)
            logger.error(
                f"API call failed (attempt {attempt}): "
                f"type={type(e).__name__} | {err[:300]}"
            )
            return None

    def _parse_vlm_json(self, raw_text: str) -> Dict:
        """
        Parse JSON from API response.
        Handles markdown fences, thinking tags, and truncated JSON.
        """
        if not raw_text:
            return self._empty_result("Empty response from API")

        text = raw_text.strip()

        # Strip thinking tags if present (Qwen-style)
        text = re.sub(r"<thinking>[\s\S]*?</thinking>\s*", "", text).strip()

        # Strip markdown fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
        if fence:
            text = fence.group(1).strip()
        else:
            # No closing fence — strip any leading ``` or ```json marker
            text = re.sub(r"^```(?:json)?\s*", "", text).strip()
            text = re.sub(r"```\s*$", "", text).strip()

        # Extract the JSON object from the first { (keeps content even if trailing text/fence)
        if not text.startswith("{"):
            brace = re.search(r"\{[\s\S]+\}", text)
            if brace:
                text = brace.group(0)
            else:
                # Truncated: no closing brace — take from first { to end
                first = text.find("{")
                if first != -1:
                    text = text[first:]

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse failed: {e}\nRaw: {text[:500]}")
            data = self._recover_partial_json(text)
            if not data:
                return self._empty_result(f"Invalid JSON from API: {str(e)[:100]}")

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
        """Best-effort recovery for truncated or malformed JSON."""
        # Clean leading non-JSON (e.g. ```json) and trailing fences
        text = text.strip()
        start = text.find("{")
        if start == -1:
            return None
        text = text[start:]

        # Trim trailing non-JSON (closing ``` etc.)
        end = text.rfind("}")
        if end != -1:
            text = text[: end + 1]

        candidates = [text]
        # Truncated: remove trailing comma and close dangling structures
        fixed = re.sub(r",\s*$", "", text)
        if fixed != text:
            candidates.append(fixed + "}")
        # Try balancing open braces/brackets
        for c in candidates:
            try:
                return json.loads(c)
            except Exception:
                pass
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

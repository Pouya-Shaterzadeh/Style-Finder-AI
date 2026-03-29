"""
Trendyol Product Search via Internal JSON API

Replaces the broken HTML scraper (Selenium + BeautifulSoup) with Trendyol's
internal search API. Benefits:
- Returns structured JSON — no HTML parsing, no CSS selector guesswork
- No bot detection / 403 blocks (JSON endpoint is less protected)
- Real product data: name, brand, price, image URL, product URL
- ~10x faster than Selenium-based scraping
"""

import requests
import time
from typing import List, Dict, Optional
from urllib.parse import quote
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    TRENDYOL_BASE_URL,
    TRENDYOL_SEARCH_URL,
    TRENDYOL_JSON_API_URL,
    MAX_SEARCH_RESULTS,
    REQUEST_DELAY,
    VISUAL_SIMILARITY_WEIGHT,
    TEXT_SIMILARITY_WEIGHT,
    DEFAULT_SIMILARITY_SCORE,
)
from src.image_processor import ImageProcessor


# Trendyol CDN prefix for product images
TRENDYOL_CDN = "https://cdn.dsmcdn.com"

# Session headers that mimic a real browser visiting Trendyol
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.trendyol.com/",
    "Origin": "https://www.trendyol.com",
    "Connection": "keep-alive",
}


class TrendyolScraper:
    """
    Fetches real Trendyol products using the internal JSON search API.

    The public.trendyol.com endpoint returns full product metadata as JSON,
    eliminating the need for HTML scraping or Selenium automation.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(_HEADERS)
        self.image_processor = ImageProcessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search_products(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
        """
        Search Trendyol for products matching the query.

        Args:
            query: Turkish search query (e.g. "Kadın Lacivert Jean")
            max_results: Max number of products to return

        Returns:
            List of product dicts with: name, brand, price, price_text,
            url, image_url, similarity_score, is_demo
        """
        print(f"Searching Trendyol (JSON API): {query}")

        products = self._search_json_api(query, max_results)

        if not products:
            print(f"⚠ JSON API returned no results for '{query}', falling back to search link")
            products = self._get_search_link_fallback(query)

        time.sleep(REQUEST_DELAY)
        return products

    def enhance_product_with_similarity(
        self,
        product: Dict,
        user_image,
        fashion_data: Dict,
    ) -> Dict:
        """
        Attach a combined visual + text similarity score to a product.

        Visual similarity (70% weight): cosine similarity between the
        user's uploaded image embedding and the product's thumbnail embedding
        (image-to-image CLIP comparison).

        Text similarity (30% weight): cosine similarity between the user's
        uploaded image embedding and the search-query text embedding
        (image-to-text CLIP comparison).

        When no product image is available the score falls back to text
        similarity only.

        Args:
            product:      Product dict from search_products()
            user_image:   Preprocessed PIL Image from user
            fashion_data: VLM output (used for query reconstruction)

        Returns:
            Product dict with 'similarity_score' and (when available)
            'visual_similarity_score' set.
        """
        if product.get("similarity_score", 0.0) > 0:
            return product  # Score already set (e.g. fallback products)

        # ── Text similarity: user image ↔ search-query text ────────────────
        query = product.get("_query", "")
        text_score = DEFAULT_SIMILARITY_SCORE
        if query and user_image is not None:
            try:
                text_score = self.image_processor.get_text_image_similarity(
                    user_image, query
                )
            except Exception as e:
                print(f"Text similarity scoring failed: {e}")

        # ── Visual similarity: user image ↔ product thumbnail ──────────────
        image_url = product.get("image_url", "")
        visual_score = None
        if image_url and user_image is not None:
            try:
                product_image = self.image_processor.load_image_from_url(image_url)
                if product_image is not None:
                    visual_score = self.image_processor.compare_images(
                        user_image, product_image
                    )
            except Exception as e:
                print(f"Visual similarity scoring failed: {e}")

        # ── Combine scores ──────────────────────────────────────────────────
        if visual_score is not None:
            product["visual_similarity_score"] = visual_score
            product["similarity_score"] = (
                VISUAL_SIMILARITY_WEIGHT * visual_score
                + TEXT_SIMILARITY_WEIGHT * text_score
            )
        else:
            # Fallback: no product image available — use text score only
            product["similarity_score"] = text_score

        return product

    # ------------------------------------------------------------------
    # JSON API
    # ------------------------------------------------------------------

    def _search_json_api(self, query: str, max_results: int) -> List[Dict]:
        """
        Call Trendyol's internal infinite-scroll product search endpoint.

        The endpoint is used by the Trendyol web app itself and returns
        structured JSON with real product data.
        """
        params = {
            "q": query,
            "pi": 1,                          # page 1
            "culture": "tr-TR",
            "userGenderId": 0,                # 0 = all genders
            "pId": 0,
            "scoringAlgorithmId": 2,
            "categoryRelevanceEnabled": "false",
            "isLegalRequirementConfirmed": "false",
            "searchStrategyType": "DEFAULT_SEARCH_STRATEGY",
        }

        try:
            resp = self.session.get(
                TRENDYOL_JSON_API_URL,
                params=params,
                timeout=15,
            )

            if resp.status_code == 200:
                data = resp.json()
                products = self._parse_json_response(data, query, max_results)
                print(f"✅ JSON API: {len(products)} products for '{query}'")
                return products

            elif resp.status_code == 403:
                print(f"⚠ Trendyol JSON API returned 403 for '{query}'")
                return []

            else:
                print(f"⚠ Trendyol JSON API: HTTP {resp.status_code} for '{query}'")
                return []

        except requests.exceptions.Timeout:
            print(f"⚠ Trendyol JSON API timeout for '{query}'")
            return []
        except Exception as e:
            print(f"⚠ Trendyol JSON API error: {e}")
            return []

    def _parse_json_response(self, data: dict, query: str, max_results: int) -> List[Dict]:
        """
        Extract product list from the Trendyol JSON API response.

        Response structure:
          data["result"]["products"] → list of product objects
          Each product:
            - name: str
            - url: str (relative, e.g. /brand/product-name-p-12345)
            - brand.name: str
            - price.sellingPrice: float
            - images: list of relative CDN paths (e.g. /ty123/.../img.jpg)
            - ratingScore.averageRating: float (optional)
        """
        raw_products = (
            data.get("result", {})
                .get("products", [])
        )

        # Some API responses nest under different keys
        if not raw_products:
            raw_products = data.get("products", [])

        products = []
        for p in raw_products[:max_results]:
            try:
                product = self._normalize_product(p, query)
                if product:
                    products.append(product)
            except Exception as e:
                print(f"Error parsing product entry: {e}")
                continue

        return products

    def _normalize_product(self, p: dict, query: str) -> Optional[Dict]:
        """Map a raw Trendyol API product dict to our internal format."""
        name = p.get("name") or p.get("productName", "")
        if not name:
            return None

        # Product URL
        relative_url = p.get("url", "")
        product_url = (
            f"{TRENDYOL_BASE_URL}{relative_url}"
            if relative_url.startswith("/")
            else relative_url
        )

        # Image URL — Trendyol images are on CDN with relative paths
        image_url = ""
        images = p.get("images", [])
        if images:
            img_path = images[0]
            if img_path.startswith("http"):
                image_url = img_path
            else:
                image_url = f"{TRENDYOL_CDN}{img_path}"

        # Price
        price_info = p.get("price", {})
        selling_price = price_info.get("sellingPrice", 0.0)
        original_price = price_info.get("originalPrice", selling_price)
        price_text = f"{selling_price:,.2f} TL"
        if original_price and original_price > selling_price:
            price_text += f"  (was {original_price:,.2f} TL)"

        # Brand
        brand = ""
        brand_info = p.get("brand", {})
        if isinstance(brand_info, dict):
            brand = brand_info.get("name", "")
        elif isinstance(brand_info, str):
            brand = brand_info

        return {
            "name": name,
            "brand": brand,
            "price": float(selling_price),
            "price_text": price_text,
            "url": product_url,
            "image_url": image_url,
            "similarity_score": 0.0,   # Will be set by enhance_product_with_similarity()
            "is_demo": False,
            "_query": query,           # Internal: used for text-image similarity scoring
        }

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _get_search_link_fallback(self, query: str) -> List[Dict]:
        """
        When the JSON API fails, return a single Trendyol search URL.
        The user can click it to see results directly on Trendyol.
        """
        encoded = quote(query)
        search_url = f"{TRENDYOL_SEARCH_URL}?q={encoded}"
        return [{
            "name": f"{query} — Trendyol'da Ara",
            "brand": "",
            "price": 0.0,
            "price_text": "",
            "url": search_url,
            "image_url": "",
            "similarity_score": 0.5,
            "is_demo": True,
            "_query": query,
        }]

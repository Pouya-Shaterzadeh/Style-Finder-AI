"""
Trendyol Product Search via Internal JSON API with HTML Fallback

Replaces the broken HTML scraper (Selenium + BeautifulSoup) with Trendyol's
internal search API. Benefits:
- Returns structured JSON — no HTML parsing, no CSS selector guesswork
- No bot detection / 403 blocks (JSON endpoint is less protected)
- Real product data: name, brand, price, image URL, product URL
- ~10x faster than Selenium-based scraping
- HTML fallback when JSON API fails (scrapes search page)
"""

import requests
import time
import re
import json
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

# HTML page headers (for search page)
_HTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
}


class TrendyolScraper:
    """
    Fetches real Trendyol products using the internal JSON search API.

    The public.trendyol.com endpoint returns full product metadata as JSON,
    eliminating the need for HTML scraping or Selenium automation.

    Falls back to HTML scraping of search page when JSON API fails.
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
            print(f"⚠ JSON API returned no results for '{query}', trying HTML fallback...")
            products = self._search_html_fallback(query, max_results)

        if not products:
            print(f"⚠ HTML fallback failed for '{query}', returning search link")
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
        Attach a similarity score to a product using text-image CLIP similarity.

        Instead of downloading each product thumbnail (slow, unreliable),
        we compare the user's uploaded image against the search query text
        using fashion-CLIP. Products from the same query get the same score.

        Args:
            product: Product dict from search_products()
            user_image: Preprocessed PIL Image from user
            fashion_data: VLM output (used for query reconstruction)

        Returns:
            Product dict with 'similarity_score' set
        """
        if product.get("similarity_score", 0.0) > 0:
            return product  # Score already set (e.g. fallback products)

        query = product.get("_query", "")
        if query and user_image is not None:
            try:
                score = self.image_processor.get_text_image_similarity(user_image, query)
                product["similarity_score"] = score
            except Exception as e:
                print(f"Similarity scoring failed: {e}")
                product["similarity_score"] = 0.5
        else:
            product["similarity_score"] = 0.5

        return product

    # ------------------------------------------------------------------
    # JSON API (Primary)
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

    # ------------------------------------------------------------------
    # HTML Fallback (Secondary)
    # ------------------------------------------------------------------

    def _search_html_fallback(self, query: str, max_results: int) -> List[Dict]:
        """
        Scrape the Trendyol search results HTML page.

        1. First visit homepage to get Cloudflare cookies (__cf_bm, __cflb)
        2. Then request search results page with those cookies
        3. Extract __SEARCH_APP_INITIAL_STATE__ JSON from the page
        4. Parse products into our standard format
        """
        try:
            # Create a fresh session for HTML scraping (separate from JSON session)
            html_session = requests.Session()
            html_session.headers.update(_HTML_HEADERS)

            # Step 1: Visit homepage to get cookies (Cloudflare challenge cookies)
            print(f"  HTML fallback: Getting cookies from homepage...")
            homepage_resp = html_session.get("https://www.trendyol.com/", timeout=10)
            if homepage_resp.status_code != 200:
                print(f"  HTML fallback: Homepage returned {homepage_resp.status_code}")
                return []

            cookies = html_session.cookies.get_dict()
            print(f"  HTML fallback: Got cookies: {list(cookies.keys())}")

            # Step 2: Request search page
            search_url = f"{TRENDYOL_SEARCH_URL}?q={quote(query)}"
            print(f"  HTML fallback: Searching {search_url}")
            search_resp = html_session.get(search_url, timeout=15)

            if search_resp.status_code == 403:
                print(f"  HTML fallback: Blocked by Cloudflare (403)")
                return []
            elif search_resp.status_code != 200:
                print(f"  HTML fallback: Search page returned {search_resp.status_code}")
                return []

            # Step 3: Extract __SEARCH_APP_INITIAL_STATE__ from HTML
            html = search_resp.text
            match = re.search(r'__SEARCH_APP_INITIAL_STATE__\s*=\s*({.*?});', html, re.DOTALL)

            if not match:
                print(f"  HTML fallback: __SEARCH_APP_INITIAL_STATE__ not found in HTML")
                # Check if it's a Cloudflare challenge page
                if "Just a moment" in html or "cf-browser-verification" in html:
                    print(f"  HTML fallback: Cloudflare challenge detected")
                return []

            print(f"  HTML fallback: Found __SEARCH_APP_INITIAL_STATE__")

            try:
                state = json.loads(match.group(1))
            except json.JSONDecodeError as e:
                print(f"  HTML fallback: Failed to parse JSON state: {e}")
                return []

            # Step 4: Parse products from state
            products_data = state.get("products", {}).get("contents", [])

            if not products_data:
                print(f"  HTML fallback: No products in state")
                return []

            print(f"  HTML fallback: Found {len(products_data)} products in HTML")

            products = []
            for p in products_data[:max_results]:
                try:
                    product = self._normalize_html_product(p, query)
                    if product:
                        products.append(product)
                except Exception as e:
                    print(f"Error parsing HTML product: {e}")
                    continue

            print(f"  HTML fallback: ✅ Parsed {len(products)} products")
            return products

        except requests.exceptions.Timeout:
            print(f"  HTML fallback: Timeout for '{query}'")
            return []
        except Exception as e:
            print(f"  HTML fallback: Error: {e}")
            return []

    def _normalize_html_product(self, p: dict, query: str) -> Optional[Dict]:
        """Map a Trendyol HTML state product to our internal format."""
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

        # Image URL
        image_url = ""
        images = p.get("images", [])
        if images:
            img_path = images[0]
            if isinstance(img_path, dict):
                img_path = img_path.get("url", "")
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
            if isinstance(img_path, dict):
                img_path = img_path.get("url", "")
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
    # Final Fallback (Search Link)
    # ------------------------------------------------------------------

    def _get_search_link_fallback(self, query: str) -> List[Dict]:
        """
        When all else fails, return a single Trendyol search URL.
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

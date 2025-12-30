"""
Trendyol.com Scraper
Handles product search and extraction from Trendyol.com
"""

import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional
import re
from urllib.parse import urljoin, quote
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    TRENDYOL_BASE_URL,
    TRENDYOL_SEARCH_URL,
    MAX_SEARCH_RESULTS,
    REQUEST_DELAY
)
from src.utils import clean_text, extract_price, validate_image_url
from src.image_processor import ImageProcessor


class TrendyolScraper:
    """Scraper for Trendyol.com product search"""
    
    def __init__(self):
        """Initialize the scraper"""
        self.base_url = TRENDYOL_BASE_URL
        self.search_url = TRENDYOL_SEARCH_URL
        self.session = requests.Session()
        self.image_processor = ImageProcessor()
        self.use_selenium = False
        self.driver = None
        
        # Set headers to mimic a real browser more convincingly
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
        })
        
        # Try to initialize Selenium as backup
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.chrome.service import Service
            
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36')
            
            # Try to use ChromeDriver from /usr/local/bin (Docker) or system PATH
            chromedriver_path = '/usr/local/bin/chromedriver'
            if os.path.exists(chromedriver_path):
                service = Service(chromedriver_path)
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                # Fallback to system PATH
                self.driver = webdriver.Chrome(options=chrome_options)
            
            self.use_selenium = True
            print("✓ Selenium initialized for Trendyol scraping")
        except Exception as e:
            print(f"⚠ Selenium not available: {e}")
            print("  Using requests with enhanced headers (may be blocked)")
    
    def search_products(self, query: str, max_results: int = MAX_SEARCH_RESULTS) -> List[Dict]:
        """
        Search for products on Trendyol
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of product dictionaries
        """
        # Build search URL
        search_url = f"{self.search_url}?q={quote(query)}"
        print(f"Searching Trendyol for: {query}")
        
        # Try Selenium first (more reliable)
        if self.use_selenium and self.driver:
            try:
                return self._search_with_selenium(search_url, max_results)
            except Exception as e:
                print(f"Selenium search failed: {e}, trying requests...")
        
        # Fall back to requests
        try:
            # First visit homepage to get cookies
            self.session.get(self.base_url, timeout=10)
            time.sleep(0.5)
            
            # Then search
            response = self.session.get(search_url, timeout=15)
            
            if response.status_code == 403:
                print("⚠ Trendyol blocked the request (403 Forbidden)")
                print("  Trendyol uses anti-bot protection.")
                print("  For production, consider using their official API or Selenium.")
                return self._get_mock_products(query, max_results)
            
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract products
            products = self._extract_products(soup, max_results)
            
            # Add delay to avoid rate limiting
            time.sleep(REQUEST_DELAY)
            
            return products
            
        except requests.exceptions.RequestException as e:
            print(f"Error searching Trendyol: {e}")
            return self._get_mock_products(query, max_results)
        except Exception as e:
            print(f"Unexpected error in search: {e}")
            return self._get_mock_products(query, max_results)
    
    def _search_with_selenium(self, search_url: str, max_results: int) -> List[Dict]:
        """Search using Selenium browser automation"""
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        print("Using Selenium for search...")
        try:
            self.driver.get(search_url)
            time.sleep(3)  # Wait for JavaScript to load
            
            # Get page source and parse
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            products = self._extract_products(soup, max_results)
            
            # If no products found via Selenium, use mock products
            if not products:
                print("Selenium found no products, using demo links...")
                # Extract query from URL
                query = search_url.split('q=')[-1].replace('%20', ' ')
                return self._get_mock_products(query, max_results)
            
            return products
        except Exception as e:
            print(f"Selenium error: {e}")
            query = search_url.split('q=')[-1].replace('%20', ' ')
            return self._get_mock_products(query, max_results)
    
    def _translate_to_turkish(self, query: str) -> str:
        """
        Translate English fashion terms to Turkish for better Trendyol search results
        
        Args:
            query: English search query (may include gender, color, item type)
            
        Returns:
            Turkish search query
        """
        query_lower = query.lower()
        
        # Gender translations (should be first in query)
        gender_map = {
            'male': 'erkek',
            'men': 'erkek',
            'man': 'erkek',
            'female': 'kadın',
            'women': 'kadın',
            'woman': 'kadın',
        }
        
        # Color translations
        color_map = {
            'white': 'beyaz',
            'black': 'siyah',
            'blue': 'mavi',
            'red': 'kırmızı',
            'green': 'yeşil',
            'brown': 'kahverengi',
            'gray': 'gri',
            'grey': 'gri',
            'pink': 'pembe',
            'yellow': 'sarı',
            'purple': 'mor',
            'orange': 'turuncu',
            'navy': 'lacivert',
            'beige': 'bej',
        }
        
        # Item type translations
        item_map = {
            'shirt': 'gömlek',
            't-shirt': 'tişört',
            'tee': 'tişört',
            'top': 'üst',
            'blouse': 'bluz',
            'pants': 'pantolon',
            'trousers': 'pantolon',
            'jeans': 'jean',
            'denim': 'jean',
            'jacket': 'ceket',
            'coat': 'mont',
            'blazer': 'blazer',
            'dress': 'elbise',
            'skirt': 'etek',
            'shoes': 'ayakkabı',
            'sneakers': 'spor ayakkabı',
            'boots': 'bot',
            'heels': 'topuklu',
            'bag': 'çanta',
            'purse': 'çanta',
            'handbag': 'el çantası',
            'backpack': 'sırt çantası',
            'hat': 'şapka',
            'cap': 'şapka',
            'sweater': 'kazak',
            'hoodie': 'kapüşonlu',
            'shorts': 'şort',
        }
        
        # Pattern/style translations
        style_map = {
            'casual': 'günlük',
            'formal': 'klasik',
            'sporty': 'spor',
            'elegant': 'şık',
            'minimalist': 'minimal',
            'solid': 'düz',
            'striped': 'çizgili',
            'patterned': 'desenli',
        }
        
        # Split query into words and translate
        words = query_lower.split()
        translated_words = []
        
        for word in words:
            # Try gender first (should be at the beginning)
            if word in gender_map:
                translated_words.append(gender_map[word])
            # Try color
            elif word in color_map:
                translated_words.append(color_map[word])
            # Try item type
            elif word in item_map:
                translated_words.append(item_map[word])
            # Try style
            elif word in style_map:
                translated_words.append(style_map[word])
            # Keep original if no translation found
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def _get_mock_products(self, query: str, max_results: int) -> List[Dict]:
        """
        Generate mock Trendyol product links for demonstration.
        In production, you would use Trendyol's official API or proper web scraping.
        Creates unique URLs for each product based on the query.
        """
        print(f"⚠ Generating demo product links for: {query}")
        
        # Translate to Turkish for better search results
        turkish_query = self._translate_to_turkish(query)
        print(f"  Translated to Turkish: {turkish_query}")
        
        # Generate realistic Trendyol search URLs with Turkish terms
        encoded_query = quote(turkish_query)
        base_search_url = f"https://www.trendyol.com/sr?q={encoded_query}"
        
        products = []
        
        # Generate ONE product per query (not multiple duplicates)
        # Each query represents a specific item, so we only need one product per item
        product = {
            'name': f"{turkish_query.title()} - Trendyol'da Ara",
            'price': None,  # Real price would come from scraping
            'brand': 'Çeşitli Markalar',
            'url': base_search_url,  # Unique URL for this specific item
            'image_url': None,
            'description': f"Trendyol'da '{turkish_query}' araması",
            'similarity_score': 0.80,  # Good score for demo products
            'is_demo': True,  # Mark as demo product
        }
        products.append(product)
        
        print(f"✓ Generated {len(products)} demo product link for: {turkish_query}")
        return products
    
    def _extract_products(self, soup: BeautifulSoup, max_results: int) -> List[Dict]:
        """
        Extract product information from search results page
        
        Args:
            soup: BeautifulSoup object of the search results page
            max_results: Maximum number of products to extract
            
        Returns:
            List of product dictionaries
        """
        products = []
        
        # Trendyol uses different selectors - try multiple approaches
        # Method 1: Look for product cards with class names
        product_cards = soup.find_all('div', class_=re.compile(r'product|item|card', re.I))
        
        # Method 2: Look for links with product URLs
        if not product_cards:
            product_cards = soup.find_all('a', href=re.compile(r'/.*-p-', re.I))
        
        # Method 3: Look for divs with data-product-id
        if not product_cards:
            product_cards = soup.find_all('div', {'data-product-id': True})
        
        # Limit results
        product_cards = product_cards[:max_results * 2]  # Get more to filter
        
        for card in product_cards:
            try:
                product = self._extract_product_info(card)
                if product and product.get('name') and product.get('url'):
                    products.append(product)
                    if len(products) >= max_results:
                        break
            except Exception as e:
                print(f"Error extracting product: {e}")
                continue
        
        return products
    
    def _extract_product_info(self, element) -> Optional[Dict]:
        """
        Extract product information from a single product element
        
        Args:
            element: BeautifulSoup element containing product info
            
        Returns:
            Product dictionary or None
        """
        product = {}
        
        # Extract product name
        name_elem = element.find(['h3', 'h2', 'span', 'div'], class_=re.compile(r'title|name|product.*name', re.I))
        if not name_elem:
            name_elem = element.find('a', class_=re.compile(r'title|name', re.I))
        if name_elem:
            product['name'] = clean_text(name_elem.get_text())
        
        # Extract product URL
        link_elem = element.find('a', href=True)
        if link_elem:
            href = link_elem.get('href', '')
            if href:
                if href.startswith('http'):
                    product['url'] = href
                else:
                    product['url'] = urljoin(self.base_url, href)
        
        # Extract price
        price_elem = element.find(['span', 'div'], class_=re.compile(r'price|fiyat', re.I))
        if price_elem:
            price_text = clean_text(price_elem.get_text())
            product['price'] = extract_price(price_text)
            product['price_text'] = price_text
        else:
            product['price'] = 0.0
            product['price_text'] = "Fiyat bilgisi yok"
        
        # Extract image URL
        img_elem = element.find('img', src=True)
        if img_elem:
            img_src = img_elem.get('src') or img_elem.get('data-src') or img_elem.get('data-lazy-src')
            if img_src:
                if img_src.startswith('http'):
                    product['image_url'] = img_src
                else:
                    product['image_url'] = urljoin(self.base_url, img_src)
        
        # Extract brand if available
        brand_elem = element.find(['span', 'div', 'a'], class_=re.compile(r'brand|marka', re.I))
        if brand_elem:
            product['brand'] = clean_text(brand_elem.get_text())
        
        # Only return if we have at least name and URL
        if product.get('name') and product.get('url'):
            # Set default values
            product.setdefault('image_url', '')
            product.setdefault('brand', '')
            product.setdefault('similarity_score', 0.0)
            
            return product
        
        return None
    
    def get_product_details(self, product_url: str) -> Optional[Dict]:
        """
        Get detailed product information from product page
        
        Args:
            product_url: URL of the product page
            
        Returns:
            Detailed product dictionary or None
        """
        try:
            response = self.session.get(product_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract additional details
            details = {}
            
            # Try to get better quality image
            img_elem = soup.find('img', class_=re.compile(r'product|main.*image', re.I))
            if img_elem:
                img_src = img_elem.get('src') or img_elem.get('data-src')
                if img_src:
                    details['image_url'] = urljoin(self.base_url, img_src) if not img_src.startswith('http') else img_src
            
            # Get description
            desc_elem = soup.find('div', class_=re.compile(r'description|aciklama', re.I))
            if desc_elem:
                details['description'] = clean_text(desc_elem.get_text())
            
            time.sleep(REQUEST_DELAY)
            return details
            
        except Exception as e:
            print(f"Error getting product details: {e}")
            return None
    
    def calculate_visual_similarity(self, user_image, product_image_url: str) -> float:
        """
        Calculate visual similarity between user image and product image
        
        Args:
            user_image: PIL Image from user upload
            product_image_url: URL of product image
            
        Returns:
            Similarity score (0-1)
        """
        try:
            product_image = self.image_processor.load_image_from_url(product_image_url)
            if product_image is None:
                return 0.0
            
            similarity = self.image_processor.compare_images(user_image, product_image)
            return similarity
            
        except Exception as e:
            print(f"Error calculating visual similarity: {e}")
            return 0.0
    
    def enhance_product_with_similarity(self, product: Dict, user_image, fashion_attributes: Dict) -> Dict:
        """
        Enhance product with similarity scores and additional info
        
        Args:
            product: Product dictionary
            user_image: User uploaded PIL Image
            fashion_attributes: Fashion attributes from VLM
            
        Returns:
            Enhanced product dictionary
        """
        # If product already has a similarity_score (e.g., demo products), preserve it
        # but still calculate visual and text scores for display
        has_existing_score = 'similarity_score' in product
        
        # Calculate visual similarity if image URL is available
        if product.get('image_url') and validate_image_url(product['image_url']):
            visual_score = self.calculate_visual_similarity(user_image, product['image_url'])
        else:
            visual_score = 0.0
        product['visual_similarity'] = visual_score
        
        # Calculate text similarity based on fashion attributes
        text_score = self._calculate_text_similarity(product, fashion_attributes)
        product['text_similarity'] = text_score
        
        # For demo products or products without images, give them a good text-based score
        if product.get('is_demo', False):
            # Demo products get a high score based on query match
            text_score = max(text_score, 0.8)  # Ensure at least 0.8 for demo products
            # Also ensure visual_score doesn't drag down the total
            visual_score = 0.3  # Give a small boost for demo products
        
        # Combined similarity score (weighted)
        from config.config import VISUAL_SIMILARITY_WEIGHT, TEXT_SIMILARITY_WEIGHT
        
        # If product already has a score (demo products), use it or the calculated one, whichever is higher
        if has_existing_score:
            calculated_score = (
                visual_score * VISUAL_SIMILARITY_WEIGHT +
                text_score * TEXT_SIMILARITY_WEIGHT
            )
            product['similarity_score'] = max(product['similarity_score'], calculated_score)
        else:
            product['similarity_score'] = (
                visual_score * VISUAL_SIMILARITY_WEIGHT +
                text_score * TEXT_SIMILARITY_WEIGHT
            )
        
        return product
    
    def _calculate_text_similarity(self, product: Dict, fashion_attributes: Dict) -> float:
        """
        Calculate text-based similarity between product and fashion attributes
        
        Args:
            product: Product dictionary
            fashion_attributes: Fashion attributes from VLM
            
        Returns:
            Text similarity score (0-1)
        """
        product_name = (product.get('name', '') + ' ' + product.get('brand', '')).lower()
        product_text = product_name
        
        # Extract keywords from fashion attributes
        fashion_keywords = []
        for item in fashion_attributes.get('items', []):
            fashion_keywords.extend([
                item.get('type', '').lower(),
                item.get('color', '').lower(),
                item.get('pattern', '').lower(),
                item.get('style', '').lower(),
            ])
        
        fashion_text = ' '.join([k for k in fashion_keywords if k and k != 'unknown'])
        
        # Simple keyword matching
        if not fashion_text:
            return 0.5  # Neutral score if no fashion data
        
        fashion_words = set(fashion_text.split())
        product_words = set(product_text.split())
        
        if not fashion_words:
            return 0.5
        
        # Calculate Jaccard similarity
        intersection = len(fashion_words & product_words)
        union = len(fashion_words | product_words)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        return min(1.0, similarity * 2)  # Boost score slightly


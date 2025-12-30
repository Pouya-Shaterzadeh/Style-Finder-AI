"""
Utility functions for Style Finder AI
"""

import re
from typing import List, Dict, Any
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean and normalize text
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Strip
    text = text.strip()
    
    return text


def extract_price(price_text: str) -> float:
    """
    Extract numeric price from text
    
    Args:
        price_text: Price string (e.g., "1.234,56 TL" or "1234.56")
        
    Returns:
        Price as float
    """
    if not price_text:
        return 0.0
    
    # Remove currency symbols and text
    price_text = re.sub(r'[^\d,.]', '', price_text)
    
    # Handle Turkish number format (1.234,56)
    if ',' in price_text and '.' in price_text:
        # Turkish format: dot for thousands, comma for decimals
        price_text = price_text.replace('.', '').replace(',', '.')
    elif ',' in price_text:
        # Could be decimal separator
        price_text = price_text.replace(',', '.')
    
    try:
        return float(price_text)
    except ValueError:
        return 0.0


def format_price(price: float, currency: str = "TL") -> str:
    """
    Format price for display
    
    Args:
        price: Price as float
        currency: Currency symbol
        
    Returns:
        Formatted price string
    """
    if price == 0:
        return "Fiyat bilgisi yok"
    
    # Format with Turkish number format
    price_str = f"{price:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    return f"{price_str} {currency}"


def build_trendyol_url(query: str) -> str:
    """
    Build Trendyol search URL from query
    
    Args:
        query: Search query string
        
    Returns:
        Trendyol search URL
    """
    from config.config import TRENDYOL_SEARCH_URL
    
    # URL encode the query
    import urllib.parse
    encoded_query = urllib.parse.quote(query)
    
    return f"{TRENDYOL_SEARCH_URL}?q={encoded_query}"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - 3] + "..."


def validate_image_url(url: str) -> bool:
    """
    Validate if URL is a valid image URL
    
    Args:
        url: URL string
        
    Returns:
        True if valid image URL
    """
    if not url:
        return False
    
    # Check if it's a valid URL
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not url_pattern.match(url):
        return False
    
    # Check for image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
    url_lower = url.lower()
    
    return any(url_lower.endswith(ext) for ext in image_extensions) or 'image' in url_lower


def merge_product_results(results_list: List[List[Dict]]) -> List[Dict]:
    """
    Merge multiple product result lists and remove duplicates
    
    Args:
        results_list: List of product result lists
        
    Returns:
        Merged and deduplicated product list
    """
    seen_urls = set()
    merged = []
    
    for results in results_list:
        for product in results:
            url = product.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                merged.append(product)
    
    return merged


def sort_products_by_score(products: List[Dict], reverse: bool = True) -> List[Dict]:
    """
    Sort products by their similarity score
    
    Args:
        products: List of product dictionaries
        reverse: If True, sort descending (highest score first)
        
    Returns:
        Sorted product list
    """
    return sorted(
        products,
        key=lambda x: x.get('similarity_score', 0.0),
        reverse=reverse
    )


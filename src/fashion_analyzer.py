"""
Main Fashion Analysis Pipeline
Orchestrates VLM analysis and Trendyol product search
"""

from typing import Dict, List, Optional, Tuple
from PIL import Image
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vlm_service import VLMService
from src.trendyol_scraper import TrendyolScraper
from src.image_processor import ImageProcessor
from src.utils import merge_product_results, sort_products_by_score
from config.config import (
    MAX_SEARCH_RESULTS,
    MIN_SIMILARITY_SCORE,
    MAX_RESULTS_DISPLAY
)


class FashionAnalyzer:
    """Main class for fashion analysis and product matching"""
    
    def __init__(self):
        """Initialize the fashion analyzer"""
        self.vlm_service = VLMService()
        self.trendyol_scraper = TrendyolScraper()
        self.image_processor = ImageProcessor()
    
    def analyze_and_find_products(
        self,
        image: Image.Image,
        max_results: int = MAX_RESULTS_DISPLAY
    ) -> Dict:
        """
        Complete pipeline: Analyze image and find matching products
        
        Args:
            image: PIL Image object from user upload
            max_results: Maximum number of products to return
            
        Returns:
            Dictionary with analysis results and products
        """
        result = {
            'success': False,
            'fashion_analysis': {},
            'products': [],
            'error': None
        }
        
        try:
            # Step 1: Preprocess image
            processed_image = self.image_processor.preprocess_image(image)
            
            # Step 2: Analyze fashion with VLM
            print("Analyzing fashion image with VLM...")
            fashion_data = self.vlm_service.analyze_fashion_image(processed_image)
            result['fashion_analysis'] = fashion_data
            
            if not fashion_data.get('items'):
                result['error'] = "No clothing items detected in the image."
                return result
            
            # Step 3: Generate search queries
            search_queries = self.vlm_service.get_search_queries(fashion_data)
            
            if not search_queries:
                result['error'] = "Could not generate search queries from analysis."
                return result
            
            print(f"Generated {len(search_queries)} search queries: {search_queries}")
            
            # Step 4: Search Trendyol for each query
            # Map each query to its corresponding item for better product naming
            query_to_item = {}
            for item in fashion_data.get('items', []):
                color = item.get('color', '')
                item_type = item.get('type', '')
                if color != 'unknown' and item_type:
                    query_key = f"{color} {item_type}"
                    query_to_item[query_key] = item
            
            all_products = []
            for query in search_queries:
                print(f"Searching Trendyol with query: {query}")
                products = self.trendyol_scraper.search_products(
                    query,
                    max_results=MAX_SEARCH_RESULTS
                )
                
                # Enhance products with similarity scores and item context
                enhanced_products = []
                item_context = query_to_item.get(query, {})
                for product in products:
                    # Update product name to be more specific to the item
                    if item_context:
                        item_type = item_context.get('type', '').title()
                        item_color = item_context.get('color', '').title()
                        if item_type and item_color:
                            product['name'] = f"{item_color} {item_type} - Trendyol'da Ara"
                    
                    enhanced = self.trendyol_scraper.enhance_product_with_similarity(
                        product,
                        processed_image,
                        fashion_data
                    )
                    enhanced_products.append(enhanced)
                
                all_products.append(enhanced_products)
            
            # Step 5: Merge and rank products
            merged_products = merge_product_results(all_products)
            ranked_products = sort_products_by_score(merged_products, reverse=True)
            
            # Step 6: Filter by minimum similarity score
            filtered_products = [
                p for p in ranked_products
                if p.get('similarity_score', 0.0) >= MIN_SIMILARITY_SCORE
            ]
            
            # Step 7: Limit results
            final_products = filtered_products[:max_results]
            
            result['products'] = final_products
            result['success'] = True
            
            print(f"Found {len(final_products)} matching products")
            
            return result
            
        except Exception as e:
            print(f"Error in fashion analysis pipeline: {e}")
            result['error'] = str(e)
            return result
    
    def analyze_single_item(
        self,
        image: Image.Image,
        item_type: str
    ) -> Dict:
        """
        Analyze and find products for a specific item type
        
        Args:
            image: PIL Image object
            item_type: Type of clothing item (e.g., "dress", "shirt")
            
        Returns:
            Dictionary with products for the specific item
        """
        # This can be used for more targeted searches
        processed_image = self.image_processor.preprocess_image(image)
        
        # Create focused fashion data
        fashion_data = {
            'items': [{'type': item_type, 'color': 'unknown', 'pattern': 'unknown', 'style': 'unknown'}],
            'overall_style': 'unknown',
            'occasion': 'unknown'
        }
        
        # Search with item type
        products = self.trendyol_scraper.search_products(item_type, max_results=MAX_SEARCH_RESULTS)
        
        # Enhance with similarity
        enhanced_products = []
        for product in products:
            enhanced = self.trendyol_scraper.enhance_product_with_similarity(
                product,
                processed_image,
                fashion_data
            )
            enhanced_products.append(enhanced)
        
        ranked_products = sort_products_by_score(enhanced_products, reverse=True)
        filtered_products = [
            p for p in ranked_products
            if p.get('similarity_score', 0.0) >= MIN_SIMILARITY_SCORE
        ]
        
        return {
            'success': True,
            'products': filtered_products[:MAX_RESULTS_DISPLAY],
            'item_type': item_type
        }
    
    def get_analysis_summary(self, fashion_data: Dict) -> str:
        """
        Generate a human-readable summary of fashion analysis
        
        Args:
            fashion_data: Fashion analysis dictionary from VLM
            
        Returns:
            Formatted summary string
        """
        if not fashion_data.get('items'):
            return "No items detected in the image."
        
        summary_parts = []
        summary_parts.append("Fashion Analysis Results:\n")
        summary_parts.append("=" * 50 + "\n")
        
        for i, item in enumerate(fashion_data['items'], 1):
            summary_parts.append(f"\nItem {i}:")
            summary_parts.append(f"  Type: {item.get('type', 'Unknown')}")
            
            if item.get('color') and item['color'] != 'unknown':
                summary_parts.append(f"  Color: {item['color']}")
            
            if item.get('pattern') and item['pattern'] != 'unknown':
                summary_parts.append(f"  Pattern: {item['pattern']}")
            
            if item.get('style') and item['style'] != 'unknown':
                summary_parts.append(f"  Style: {item['style']}")
            
            if item.get('description'):
                summary_parts.append(f"  Description: {item['description']}")
        
        if fashion_data.get('overall_style') and fashion_data['overall_style'] != 'unknown':
            summary_parts.append(f"\nOverall Style: {fashion_data['overall_style']}")
        
        if fashion_data.get('occasion') and fashion_data['occasion'] != 'unknown':
            summary_parts.append(f"Occasion: {fashion_data['occasion']}")
        
        return "\n".join(summary_parts)


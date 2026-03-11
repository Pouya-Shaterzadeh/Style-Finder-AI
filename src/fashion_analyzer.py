"""
Main Fashion Analysis Pipeline
Orchestrates VLM analysis, Turkish query generation, and Trendyol product search.
"""

from typing import Dict, List
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vlm_service import VLMService
from src.trendyol_scraper import TrendyolScraper
from src.image_processor import ImageProcessor
from src.utils import merge_product_results, sort_products_by_score
from config.config import (
    MAX_SEARCH_RESULTS,
    MIN_SIMILARITY_SCORE,
    MAX_RESULTS_DISPLAY,
)


class FashionAnalyzer:
    """Main class for fashion analysis and product matching."""

    def __init__(self):
        self.vlm_service = VLMService()
        self.trendyol_scraper = TrendyolScraper()
        self.image_processor = ImageProcessor()

    def analyze_and_find_products(
        self,
        image: Image.Image,
        max_results: int = MAX_RESULTS_DISPLAY,
    ) -> Dict:
        """
        Full pipeline: analyze image → generate queries → search Trendyol → rank.

        Args:
            image: User-uploaded PIL Image
            max_results: Max number of products to display

        Returns:
            {
              'success': bool,
              'fashion_analysis': dict,   # VLM output
              'products': list[dict],     # Ranked product list
              'error': str | None,
            }
        """
        result = {
            'success': False,
            'fashion_analysis': {},
            'products': [],
            'error': None,
        }

        try:
            # Step 1: Preprocess image
            processed_image = self.image_processor.preprocess_image(image)

            # Step 2: Analyze fashion with GPT OSS 120B
            print("Analyzing fashion image with GPT OSS 120B...")
            fashion_data = self.vlm_service.analyze_fashion_image(processed_image)
            result['fashion_analysis'] = fashion_data

            if fashion_data.get('error') and not fashion_data.get('items'):
                result['error'] = fashion_data['error']
                return result

            if not fashion_data.get('items'):
                result['error'] = (
                    "No clothing items were detected. "
                    "Please upload a clearer photo with good lighting where clothing is clearly visible."
                )
                return result

            # Step 3: Generate Turkish search queries from VLM output
            search_queries = self.vlm_service.get_search_queries(fashion_data)

            if not search_queries:
                result['error'] = (
                    "Could not generate search queries. "
                    "The detected items may not have Turkish translations yet."
                )
                return result

            print(f"Generated {len(search_queries)} queries: {search_queries}")

            # Step 4: Search Trendyol for each query
            all_product_groups = []
            for query in search_queries:
                print(f"Searching: {query}")
                products = self.trendyol_scraper.search_products(
                    query,
                    max_results=MAX_SEARCH_RESULTS,
                )

                # Attach similarity scores using fashion-CLIP (text vs image)
                enhanced = []
                for product in products:
                    scored = self.trendyol_scraper.enhance_product_with_similarity(
                        product, processed_image, fashion_data
                    )
                    enhanced.append(scored)

                all_product_groups.append(enhanced)

            # Step 5: Merge, deduplicate, and rank
            merged = merge_product_results(all_product_groups)
            ranked = sort_products_by_score(merged, reverse=True)

            # Step 6: Filter by minimum score and cap results
            filtered = [
                p for p in ranked
                if p.get('similarity_score', 0.0) >= MIN_SIMILARITY_SCORE
            ]
            result['products'] = filtered[:max_results]
            result['success'] = True

            print(f"Returning {len(result['products'])} products")
            return result

        except Exception as e:
            print(f"Pipeline error: {e}")
            result['error'] = str(e)
            return result

    def get_analysis_summary(self, fashion_data: Dict) -> str:
        """Human-readable summary of detected fashion items."""
        if not fashion_data.get('items'):
            return "No items detected."

        lines = ["Fashion Analysis:\n" + "=" * 40]
        for i, item in enumerate(fashion_data['items'], 1):
            lines.append(f"\nItem {i}: {item.get('type', '?').title()}")
            if item.get('color') and item['color'] != 'unknown':
                lines.append(f"  Color: {item['color'].title()}")
            if item.get('pattern') and item['pattern'] not in ('solid', 'unknown'):
                lines.append(f"  Pattern: {item['pattern']}")
            if item.get('material') and item['material'] != 'unknown':
                lines.append(f"  Material: {item['material']}")
            if item.get('description'):
                lines.append(f"  Note: {item['description']}")

        if fashion_data.get('overall_style') not in ('unknown', None):
            lines.append(f"\nStyle: {fashion_data['overall_style'].title()}")
        if fashion_data.get('occasion') not in ('unknown', None):
            lines.append(f"Occasion: {fashion_data['occasion'].title()}")

        return "\n".join(lines)

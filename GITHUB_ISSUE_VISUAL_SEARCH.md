# GitHub Issue: Add Visual Similarity Search Feature

Copy the content below and paste it when creating a new issue on GitHub:
https://github.com/Pouya-Shaterzadeh/Style-Finder-AI/issues/new

---

## Feature Request: Visual Similarity Search

### Description
Implement a visual similarity search feature using CLIP (Contrastive Language-Image Pre-training) embeddings to compare uploaded images with product images, finding visually similar items based on shape, color, and style - similar to Google's "Search by Image" feature.

### How Visual Search Should Work

1. **Image Embedding**: Your uploaded image is converted to a CLIP embedding (vector representation)
2. **Product Embedding**: Each product image is also converted to a CLIP embedding
3. **Similarity Calculation**: Cosine similarity compares embeddings to find visually similar items
4. **Ranking**: Results are ranked by combined visual (70%) and text (30%) similarity

### Technical Details

- Use CLIP model (`openai/clip-vit-base-patch32`) for generating embeddings
- Calculate cosine similarity between user image and product image embeddings
- Combine visual similarity (70% weight) with text similarity (30% weight) for final ranking
- Filter products by minimum similarity threshold
- Display similarity scores in product cards

### Benefits

- More accurate product matching based on visual appearance
- Better results for users who want items that look similar to their uploaded image
- Enhanced user experience with Google-like "Search by Image" functionality
- Improved product discovery beyond just text-based search

### Implementation Notes

- This feature was previously implemented but reverted. The code structure exists in the codebase and can be re-enabled.
- Ensure proper handling of products without images (fallback to text-only similarity)
- Consider caching embeddings for performance optimization
- Add visual similarity scores to the product display

### Related Files
- `src/fashion_analyzer.py` - Main analysis pipeline
- `src/image_processor.py` - Image processing and CLIP embeddings
- `src/trendyol_scraper.py` - Product search and enhancement
- `config/config.py` - Similarity weights configuration

### Labels
- `enhancement`
- `feature-request`
- `visual-search`


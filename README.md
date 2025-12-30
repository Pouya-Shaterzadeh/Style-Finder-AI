<img width="1902" height="1032" alt="Screenshot from 2025-12-30 04-26-58" src="https://github.com/user-attachments/assets/fb151533-ab1b-4784-9acb-cffb131d1392" />
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





<img width="1820" height="1008" alt="Screenshot from 2025-12-30 04-32-01" src="https://github.com/user-attachments/assets/0300db37-c3fc-4b7f-8625-aebf3f1e7b59" />

# Style Finder AI: Computer Vision-Based Fashion Analysis

A sophisticated fashion analysis application that uses Vision Language Models (VLMs) to analyze fashion images and find matching products on Trendyol.com for the Turkish market.

## Features

- **AI-Powered Fashion Analysis**: Uses BLIP Vision Language Model to extract detailed fashion attributes from images
- **Smart Color Detection**: Advanced color-item matching algorithm for accurate outfit analysis
- **High-Accuracy Product Matching**: Multi-stage matching algorithm combining VLM attributes and visual similarity
- **Trendyol Integration**: Real-time product search on Trendyol.com with direct purchase links
- **Rich UI/UX**: Modern, responsive interface with smooth animations and intuitive design
- **Multi-Item Support**: Analyzes entire outfits and finds all clothing pieces
- **Gender Detection**: Automatically detects and includes gender in product searches
- **Image-Specific Style Tips**: Provides personalized styling advice based on detected outfit

## How It Works

1. **Upload Your Image**: User uploads a clear fashion image
2. **AI Agent Analysis**: BLIP Vision Language Model acts as a Fashion Analyzer AI Agent, extracting clothing items, colors, patterns, styles, and gender
3. **Smart Matching**: System translates queries to Turkish and searches Trendyol.com using AI-extracted attributes
4. **Visual Matching**: CLIP embeddings compare uploaded image with product images for visual similarity
5. **Result Ranking**: Products ranked by combined text and visual similarity scores
6. **Display**: Matched products shown with similarity scores and direct Trendyol links

## Technology Stack

- **Vision Language Model**: BLIP (Salesforce/blip-image-captioning-base) via Hugging Face Transformers
- **Framework**: Gradio with custom CSS
- **Image Processing**: PIL, OpenCV, CLIP embeddings
- **Web Scraping**: BeautifulSoup4, Selenium (with fallback to demo products)
- **Vector Search**: FAISS, scikit-learn
- **Deployment**: Docker on Hugging Face Spaces

## Usage

Simply upload a fashion image and click "Analyze & Find Products". The system will:
1. Analyze the image to extract fashion attributes (items, colors, style, gender)
2. Generate image-specific style tips
3. Search Trendyol.com for matching products in Turkish
4. Rank products by similarity
5. Display results with direct purchase links

## Tips for Best Results

- Upload clear, well-lit images
- Ensure clothing items are clearly visible
- Images with single outfits work best
- Full-body or upper-body fashion photos work best
- The system automatically detects gender and includes it in searches

## Model Information

- **VLM Model**: BLIP (Salesforce/blip-image-captioning-base)
- **AI Agent Role**: Fashion & Style Analyzer
- **Model Size**: ~1GB (downloads automatically on first run)
- **Deployment**: Local model loading on Hugging Face Spaces

## Configuration

The app uses local model loading by default (recommended for Hugging Face Spaces). Key settings can be adjusted in `config/config.py`:

- Model selection (BLIP, Moondream, or LLaVA)
- Similarity weights (visual vs text)
- Minimum similarity threshold
- Maximum search results

## Environment Variables

Optional: Add `HF_API_TOKEN` as a Secret in your Space settings for enhanced model access:
- Go to Space → Settings → Variables and secrets → Add secret
- Name: `HF_API_TOKEN`
- Value: Your Hugging Face API token

## License

MIT License

## Author

Deployed by [PouyaDevA1](https://huggingface.co/PouyaDevA1)

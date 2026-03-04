---
title: Style Finder AI
emoji: 👗
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "6.2.0"
app_file: app.py
pinned: false
---

# Style Finder AI — Fashion Analysis & Product Discovery

Upload a fashion photo → get AI analysis + matching products from Trendyol.

## How It Works

1. **Upload** a clear fashion photo (outfit, street style, lookbook, etc.)
2. **Groq Llama 4 Maverick** analyzes the image — detects clothing items, colors, patterns, materials, fit, gender, and overall style. Returns structured JSON in a single API call.
3. **Turkish search queries** are generated from the analysis (e.g. "Kadın Lacivert Slim Jean")
4. **Trendyol's internal JSON API** is queried — real listings with prices, images, and product pages
5. **fashion-CLIP** (trained on 800K+ fashion image-text pairs) scores each result by text-image cosine similarity
6. **Results** are displayed as clickable product cards — click any card to open the Trendyol listing

## Technology Stack

| Layer | Model / Tool |
|---|---|
| Vision Language Model | Groq Llama 4 Maverick 17B (meta-llama/llama-4-maverick-17b-128e-instruct) |
| Fashion Similarity | patrickjohncyh/fashion-clip |
| Product Search | Trendyol Internal JSON API |
| UI Framework | Gradio 6 |
| Hosting | Hugging Face Spaces (CPU) |

## Environment Secrets (HF Spaces Settings → Secrets)

| Secret | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | Free key from [console.groq.com](https://console.groq.com) |
| `HF_API_TOKEN` | Optional | For faster fashion-CLIP model download |

## Tips for Best Results

- Use clear, well-lit photos
- Full-body or upper-body shots work best
- Single outfit per image gives more accurate results
- The system automatically detects gender and includes it in search queries

## License

MIT — by [PouyaDevA1](https://huggingface.co/PouyaDevA1)

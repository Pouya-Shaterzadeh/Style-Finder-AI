# Using Hugging Face Inference API (Free - No Local Model Download!)

## Overview

This application now uses **Hugging Face Inference API** instead of downloading models locally. This means:
- ✅ **No 7GB model download required**
- ✅ **Free to use** (with rate limits)
- ✅ **Faster setup** - just run the app!
- ✅ **Always uses latest model version**

## How It Works

The app sends your fashion images to Hugging Face's cloud servers where the LLaVA model runs. Results are returned instantly without any local model storage.

## Getting Started

### Option 1: Use Without API Token (Public Access)
- Works immediately - no setup needed!
- Uses public Inference API endpoints
- Subject to rate limits (slower during peak times)

### Option 2: Use With API Token (Recommended)
1. Create a free Hugging Face account: https://huggingface.co/join
2. Get your API token: https://huggingface.co/settings/tokens
3. Create a new token with "Read" permissions
4. Set it as an environment variable:
   ```bash
   export HF_API_TOKEN="your_token_here"
   ```
   Or add to your `.env` file:
   ```
   HF_API_TOKEN=your_token_here
   ```

## Benefits of Using API Token

- Higher rate limits
- Priority access during peak times
- More reliable service
- Still completely FREE!

## Rate Limits

**Without Token:**
- ~30 requests/hour (approximate)

**With Free Token:**
- ~1000 requests/month (generous for personal use)

## Troubleshooting

### "Model is loading" Error
- First request to a model may take 30-60 seconds
- The app automatically retries once
- Subsequent requests are much faster

### Rate Limit Exceeded
- Wait a few minutes and try again
- Get a free API token for higher limits
- Consider upgrading to Pro plan for production use

### Network Issues
- Check your internet connection
- Ensure you can access huggingface.co
- Try again - the API is very reliable

## Alternative: Local Model (If Needed)

If you prefer to run models locally (no internet needed, unlimited requests), you can modify `config/config.py`:

```python
USE_INFERENCE_API = False  # Load model locally
```

This will download the ~7GB LLaVA model on first run.

## Cost

**Completely FREE** for:
- Personal use
- Development
- Testing
- Small-scale production

For high-volume production use, consider Hugging Face Pro plans.


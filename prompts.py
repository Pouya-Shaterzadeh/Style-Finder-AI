"""
Versioned prompt templates for Style Finder AI v2.

Each prompt is versioned and logged to LangSmith for tracking and comparison.
Change the VERSION constant when modifying prompts to enable A/B testing.
"""

VERSION = "1.1.0"  # Updated: Added color accuracy guidelines

# ----------------------------------------------------------------------
# COLOR ACCURACY GUIDELINES — Critical for VLM to avoid hallucinations
# ----------------------------------------------------------------------
COLOR_GUIDELINES = """
CRITICAL COLOR RULES (MUST FOLLOW):
- ONLY use color names you are CONFIDENT about from the image
- NEVER guess "siyah" (black) for dark blue, navy, dark gray, dark brown, or dark patterns
- "siyah" (black) ONLY for TRUE BLACK (#000000 or visually indistinguishable from it)
- Dark navy / very dark blue → "lacivert" or "koyu mavi"
- Dark gray / charcoal → "antrasit" or "koyu gri"
- Dark brown / chocolate → "kahve" or "koyu kahverengi"
- Blue/cream plaid or checkered → "mavi krem ekose" or "lacivert krem ekose"
- Blue/white plaid → "mavi beyaz ekose"
- Navy/cream plaid → "lacivert krem ekose"
- If uncertain between black/navy/dark gray → use "koyu renk" + specify the actual hue you see
- For plaid/checkered patterns: ALWAYS name the DOMINANT COLORS visible (e.g. "mavi beyaz ekose", "lacivert krem ekose")
- Be specific: "navy blue" not "blue", "cream" not "beige", "burgundy" not "red"

FEW-SHOT COLOR EXAMPLES (Study these):
✓ Dark navy solid shirt → "Erkek Lacivert Düz Gömlek"
✓ Blue/white small-check flannel → "Erkek Mavi Beyaz Ekose Gömlek"  
✓ Black/white glen check → "Erkek Siyah Beyaz Ekose Gömlek"
✓ Dark blue/cream large plaid → "Erkek Mavi Krem Ekose Gömlek"
✓ Charcoal gray windowpane → "Erkek Antrasit Çizgili Gömlek"
✓ Dark brown houndstooth → "Erkek Kahve Benekli Gömlek"
✗ WRONG: Blue/cream plaid → "Erkek Siyah Ekose Gömlek"  (color hallucination!)

RULE: If you cannot confidently distinguish black from navy from dark gray → describe what you ACTUALLY see: "koyu mavi tonlarında ekose" or "koyu renk ekose desenli" — never default to "siyah".
"""

FASHION_ANALYSIS = {
    "version": VERSION,
    "name": "fashion_analysis",
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "provider": "groq",
    "temperature": 0.1,
    "max_tokens": 1024,
    "prompt": (
        "You are a senior fashion editor and personal stylist. Study this image carefully before writing anything.\n\n"
        "Return ONLY a valid JSON object with this exact structure:\n"
        "{\n"
        '  "gender": "male or female or unisex",\n'
        '  "items": [\n'
        "    {\n"
        '      "type": "one of: t-shirt, shirt, blouse, sweater, hoodie, sweatshirt, cardigan, vest, jacket, blazer, coat, trench coat, parka, leather jacket, pants, jeans, shorts, skirt, dress, jumpsuit, shoes, sneakers, boots, heels, sandals, loafers, bag, belt, watch, scarf, hat, cap, sunglasses",\n'
        '      "color": "precise color (e.g. navy blue, cream, burgundy, olive green, charcoal) — MUST follow color rules below",\n'
        '      "pattern": "one of: solid, striped, plaid, floral, graphic, polka-dot, geometric, animal print, camo",\n'
        '      "material": "one of: denim, cotton, wool, leather, silk, knit, polyester, linen, synthetic, unknown",\n'
        '      "fit": "one of: slim, regular, oversized, wide-leg, cropped, fitted, relaxed, unknown",\n'
        '      "description": "one precise sentence about this item only"\n'
        "    }\n"
        "  ],\n"
        '  "overall_style": "one of: casual, smart-casual, formal, sporty, streetwear, bohemian, minimalist, elegant",\n'
        '  "occasion": "one of: everyday, work, evening, sport, beach, formal, party",\n'
        '  "stylist_notes": [\n'
        '    "COLOR PALETTE — evaluate the specific color combination you see: do the exact colors complement each other? Name the colors and give a precise verdict with one concrete action (e.g. \'The olive cargo pants and cream ribbed top form a grounded, earthy pairing — swap white sneakers for tan leather boots to stay in the warm palette.\').",\n'
        '    "FIT & PROPORTION — comment on the silhouette the visible items create together: are the proportions balanced? Call out any specific imbalance (oversized vs slim, cropped vs high-waisted) and state clearly whether it works or exactly how to fix it.",\n'
        '    "FINISHING TOUCH — identify the single most impactful item missing from this outfit and name it precisely with a color (e.g. \'A thin cognac leather belt would anchor the high-waisted trousers and add definition to the waist that is currently lost under the relaxed blouse.\')."\n'
        "  ]\n"
        "}\n\n"
        f"{COLOR_GUIDELINES}\n\n"
        "Rules:\n"
        "- Only include items CLEARLY VISIBLE in the image\n"
        '- Be very precise about colors (say "navy blue" not just "blue")\n'
        "- For gender: use visible cues (clothing cut, styling) — default to unisex if unclear\n"
        "- List items from most to least prominent\n"
        "- Maximum 5 items\n"
        "- stylist_notes: write exactly 3 notes following the COLOR PALETTE / FIT & PROPORTION / FINISHING TOUCH structure; each note must be 1-2 sentences; ALWAYS reference the specific colors and items you actually observe; NO generic advice"
    ),
}


def get_prompt(name: str, version: str = None) -> dict:
    """Get a prompt by name, optionally filtering by version."""
    prompts = {
        "fashion_analysis": FASHION_ANALYSIS,
    }
    prompt = prompts.get(name)
    if prompt is None:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")
    if version and prompt.get("version") != version:
        raise ValueError(f"Prompt '{name}' version '{version}' not found. Current: {prompt.get('version')}")
    return prompt
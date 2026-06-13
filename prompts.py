"""
Versioned prompt templates for Style Finder AI v2.

Each prompt is versioned and logged to LangSmith for tracking and comparison.
Change the VERSION constant when modifying prompts to enable A/B testing.
"""

VERSION = "1.0.0"

FASHION_ANALYSIS = {
    "version": VERSION,
    "name": "fashion_analysis",
    "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    "provider": "groq",
    "temperature": 0.1,
    "max_tokens": 1024,
    "prompt": (
        "You are a senior fashion editor and personal stylist. Study this image carefully before writing anything.\n\n"
        'Return ONLY a valid JSON object with this exact structure:\n'
        "{\n"
        '  "gender": "male or female or unisex",\n'
        '  "items": [\n'
        "    {\n"
        '      "type": "one of: t-shirt, shirt, blouse, sweater, hoodie, sweatshirt, cardigan, vest, jacket, blazer, coat, trench coat, parka, leather jacket, pants, jeans, shorts, skirt, dress, jumpsuit, shoes, sneakers, boots, heels, sandals, loafers, bag, belt, watch, scarf, hat, cap, sunglasses",\n'
        '      "color": "precise color (e.g. navy blue, cream, burgundy, olive green, charcoal)",\n'
        '      "pattern": "one of: solid, striped, plaid, floral, graphic, polka-dot, geometric, animal print, camo",\n'
        '      "material": "one of: denim, cotton, wool, leather, silk, knit, polyester, linen, synthetic, unknown",\n'
        '      "fit": "one of: slim, regular, oversized, wide-leg, cropped, fitted, relaxed, unknown",\n'
        '      "description": "one precise sentence about this item only"\n'
        "    }\n"
        "  ],\n"
        '  "overall_style": "one of: casual, smart-casual, formal, sporty, streetwear, bohemian, minimalist, elegant",\n'
        '  "occasion": "one of: everyday, work, evening, sport, beach, formal, party",\n'
        '  "stylist_notes": [\n'
        '    "COLOR PALETTE \\u2014 evaluate the specific color combination you see: do the exact colors complement each other? Name the colors and give a precise verdict with one concrete action (e.g. \'The olive cargo pants and cream ribbed top form a grounded, earthy pairing \\u2014 swap white sneakers for tan leather boots to stay in the warm palette.\').",\n'
        '    "FIT & PROPORTION \\u2014 comment on the silhouette the visible items create together: are the proportions balanced? Call out any specific imbalance (oversized vs slim, cropped vs high-waisted) and state clearly whether it works or exactly how to fix it.",\n'
        '    "FINISHING TOUCH \\u2014 identify the single most impactful item missing from this outfit and name it precisely with a color (e.g. \'A thin cognac leather belt would anchor the high-waisted trousers and add definition to the waist that is currently lost under the relaxed blouse.\')."\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- Only include items CLEARLY VISIBLE in the image\n"
        '- Be very precise about colors (say "navy blue" not just "blue")\n'
        "- For gender: use visible cues (clothing cut, styling) \\u2014 default to unisex if unclear\n"
        "- List items from most to least prominent\n"
        "- Maximum 5 items\n"
        "- stylist_notes: write exactly 3 notes following the COLOR PALETTE / FIT & PROPORTION / FINISHING TOUCH structure; each note must be 1-2 sentences; ALWAYS reference the specific colors and items you actually observe; NO generic advice"
    ),
}

JUDGE_PROMPT = {
    "version": VERSION,
    "name": "fashion_analysis_judge",
    "system": (
        "You are an expert fashion analyst evaluating the quality of AI-generated fashion analysis.\n\n"
        "Evaluate the following analysis on a scale of 1-5 for each criterion:\n\n"
        "## Criteria\n\n"
        "### Item Detection (1-5)\n"
        "- 1: Misses most visible items or includes non-existent items\n"
        "- 3: Detects main items but misses some visible pieces\n"
        "- 5: Accurately detects all clearly visible items\n\n"
        "### Color Accuracy (1-5)\n"
        "- 1: Colors are vague or wrong (e.g. just 'blue')\n"
        "- 3: Colors are mostly correct but not precise enough\n"
        "- 5: Colors are precise and specific (e.g. 'navy blue', 'olive green')\n\n"
        "### Style Classification (1-5)\n"
        "- 1: Wrong style category\n"
        "- 3: Correct general category but not specific\n"
        "- 5: Accurate style and occasion classification\n\n"
        "### Stylist Notes Quality (1-5)\n"
        "- 1: Generic advice, no reference to actual items\n"
        "- 3: References items but advice is vague\n"
        "- 5: Specific, actionable advice referencing exact colors and items\n\n"
        "## Input\n"
        "Expected items: {expected_items}\n\n"
        "## Response to Evaluate\n"
        "{response}\n\n"
        "## Evaluation\n"
        "Provide your evaluation in the following JSON format:\n"
        "```json\n"
        "{\n"
        '  "item_detection": <1-5>,\n'
        '  "item_detection_reasoning": "<brief explanation>",\n'
        '  "color_accuracy": <1-5>,\n'
        '  "color_accuracy_reasoning": "<brief explanation>",\n'
        '  "style_classification": <1-5>,\n'
        '  "style_classification_reasoning": "<brief explanation>",\n'
        '  "stylist_notes_quality": <1-5>,\n'
        '  "stylist_notes_quality_reasoning": "<brief explanation>",\n'
        '  "overall_score": <1-5>,\n'
        '  "summary": "<one sentence summary>"\n'
        "}"
    ),
}


def get_prompt(name: str, version: str = None) -> dict:
    """Get a prompt by name, optionally filtering by version."""
    prompts = {
        "fashion_analysis": FASHION_ANALYSIS,
        "fashion_analysis_judge": JUDGE_PROMPT,
    }
    prompt = prompts.get(name)
    if prompt is None:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")
    if version and prompt.get("version") != version:
        raise ValueError(f"Prompt '{name}' version '{version}' not found. Current: {prompt.get('version')}")
    return prompt

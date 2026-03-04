"""
Style Finder AI - Main Application
Computer Vision-Based Fashion Analysis with Trendyol Integration
"""

import gradio as gr
from PIL import Image
import sys
import os
from typing import Tuple, Optional

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.fashion_analyzer import FashionAnalyzer
from config.config import MAX_RESULTS_DISPLAY

# Initialize the analyzer
analyzer = FashionAnalyzer()

# Load custom CSS
CSS_PATH = os.path.join(os.path.dirname(__file__), "static", "css", "custom.css")
custom_css = ""
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, 'r') as f:
        custom_css = f.read()


# ---------------------------------------------------------------------------
# Color dot helper
# ---------------------------------------------------------------------------

_COLOR_CSS: dict = {
    "black": "#111827", "white": "#f9fafb", "gray": "#6b7280", "grey": "#6b7280",
    "navy": "#1e3a5f", "navy blue": "#1e3a5f", "blue": "#3b82f6",
    "light blue": "#93c5fd", "dark blue": "#1e40af", "royal blue": "#2563eb",
    "red": "#ef4444", "dark red": "#991b1b", "burgundy": "#7f1d1d",
    "maroon": "#7f1d1d", "wine": "#7f1d1d", "green": "#22c55e",
    "dark green": "#166534", "olive": "#7c6c2f", "olive green": "#7c6c2f",
    "khaki": "#b5a642", "mint": "#6ee7b7", "mint green": "#6ee7b7",
    "brown": "#92400e", "dark brown": "#451a03", "tan": "#d4a96a",
    "camel": "#d4a96a", "beige": "#e8d5b0", "cream": "#fef3c7",
    "off-white": "#f5f0e8", "ivory": "#f5f0e8", "yellow": "#eab308",
    "mustard": "#b45309", "orange": "#f97316", "pink": "#f9a8d4",
    "hot pink": "#ec4899", "fuchsia": "#d946ef", "rose": "#fb7185",
    "purple": "#8b5cf6", "lavender": "#c4b5fd", "lilac": "#c084fc",
    "violet": "#7c3aed", "silver": "#9ca3af", "gold": "#d97706",
    "metallic": "#94a3b8", "denim": "#3b82f6", "indigo": "#4f46e5",
    "coral": "#f87171", "teal": "#0d9488", "turquoise": "#06b6d4",
    "charcoal": "#374151",
}

_LIGHT_COLORS = {"#f9fafb", "#fef3c7", "#f5f0e8", "#e8d5b0", "#93c5fd", "#c4b5fd", "#f9a8d4", "#6ee7b7"}


def _color_dot(color_name: str) -> str:
    """Return an HTML span with a small color swatch circle."""
    key = color_name.lower().strip()
    css = _COLOR_CSS.get(key, "")
    if not css:
        for k, v in _COLOR_CSS.items():
            if k in key:
                css = v
                break
    if not css:
        return ""
    border = " border: 1.5px solid rgba(0,0,0,.18);" if css in _LIGHT_COLORS else ""
    return f'<span class="sf-color-dot" style="background:{css};{border}" title="{color_name.title()}"></span>'


# ---------------------------------------------------------------------------
# HTML rendering — analysis panel
# ---------------------------------------------------------------------------

def _render_item_chips(fashion_data: dict) -> str:
    """Render item chip HTML for the Outfit Breakdown panel."""
    html_parts = []
    for item in fashion_data.get('items', []):
        item_type  = item.get("type", "Unknown").title()
        raw_color  = item.get("color", "")
        item_color = raw_color.title() if raw_color not in ("unknown", "", None) else ""
        item_display = f"{item_color} {item_type}".strip() if item_color else item_type

        extra_parts = []
        pattern     = item.get("pattern", "")
        material    = item.get("material", "")
        fit         = item.get("fit", "")
        description = item.get("description", "")
        if pattern  and pattern  not in ("solid",   "unknown", ""):
            extra_parts.append(pattern.title())
        if material and material not in ("unknown", ""):
            extra_parts.append(material.title())
        if fit      and fit      not in ("unknown", ""):
            extra_parts.append(fit.title())

        dot_html  = _color_dot(raw_color) if raw_color else ""
        tags_html = "".join(f'<span class="sf-chip-tag">{p}</span>' for p in extra_parts)

        html_parts.append(f"""
        <div class="sf-item-chip">
            <div class="sf-chip-header">
                {dot_html}
                <span class="sf-item-chip-name">{item_display}</span>
            </div>
            {f'<div class="sf-chip-tags">{tags_html}</div>' if tags_html else ''}
            {f'<div class="sf-item-chip-desc">{description}</div>' if description else ''}
        </div>
        """)
    return "".join(html_parts)


def format_analysis_html(result: dict) -> str:
    """Render the Outfit Breakdown + Stylist Notes panel (left column)."""
    if not result.get('success'):
        return ""

    fashion_data = result.get('fashion_analysis', {})
    items        = fashion_data.get('items', [])
    style_tips   = fashion_data.get('stylist_notes', [])

    if not items and not style_tips:
        return ""

    html_parts = ['<div class="sf-analysis-panel">']
    html_parts.append('<h3 class="sf-analysis-title">Outfit Breakdown</h3>')
    html_parts.append(_render_item_chips(fashion_data))

    if style_tips:
        html_parts.append('<div class="sf-tips-section">')
        html_parts.append('<h4 class="sf-tips-title">Stylist Notes</h4>')
        for tip in style_tips:
            if tip and tip.strip():
                html_parts.append(f'<div class="sf-tip-item">{tip}</div>')
        html_parts.append('</div>')

    html_parts.append('</div>')
    return ''.join(html_parts)


# ---------------------------------------------------------------------------
# HTML rendering — product grid
# ---------------------------------------------------------------------------

def format_product_card_compact(product: dict) -> str:
    """Format a product as a compact HTML card using CSS classes."""
    name             = product.get('name', 'Unknown Product')
    product_url      = product.get('url', '#')
    image_url        = product.get('image_url', '')
    brand            = product.get('brand', '')
    price_text       = product.get('price_text', '')
    similarity_score = product.get('similarity_score', 0.0)
    similarity_percent = int(similarity_score * 100)
    is_demo          = product.get('is_demo', False)

    # Clean display name — no truncation
    display_name = name.replace(" — Trendyol'da Ara", "").replace(" - Trendyol'da Ara", "").strip()

    img_html   = f'<img src="{image_url}" alt="{display_name}" class="sf-card-img" onerror="this.style.display=\'none\'">' if image_url else ''
    badge_class = "sf-badge sf-badge-real" if not is_demo else "sf-badge sf-badge-demo"
    badge_label = f"{similarity_percent}% Match" if not is_demo else "Search Link"
    brand_html  = f'<div class="sf-card-brand">{brand}</div>' if brand else ''
    price_html  = f'<div class="sf-card-price">{price_text}</div>' if price_text else ''

    return f"""
    <div class="sf-card fade-in">
        <a href="{product_url}" target="_blank" rel="noopener noreferrer">
            {img_html}
            <div class="sf-card-body">
                {brand_html}
                <div class="sf-card-name">{display_name}</div>
                {price_html}
                <div class="sf-card-footer">
                    <span class="{badge_class}">{badge_label}</span>
                    <span class="sf-card-cta">View →</span>
                </div>
            </div>
        </a>
    </div>
    """


def format_results_html(result: dict) -> str:
    """Format analysis results as HTML — two-column sf-results-grid layout."""
    if not result.get('success'):
        error = result.get('error', 'Unknown error occurred')
        return f"""
        <div class="sf-error-panel">
            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: #dc2626;">
                Unable to Analyze Image
            </div>
            <div style="font-size: 1rem; line-height: 1.6;">{error}</div>
            <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #fecaca; font-size: 0.95rem;">
                <strong>Tips for better results:</strong>
                <ul style="margin: 0.75rem 0 0 1.5rem; line-height: 1.8;">
                    <li>Use clear, well-lit images</li>
                    <li>Ensure clothing items are clearly visible</li>
                    <li>Full-body or upper-body photos work best</li>
                    <li>Avoid images where the person is too far from the camera</li>
                </ul>
            </div>
        </div>
        """

    fashion_data = result.get('fashion_analysis', {})
    products     = result.get('products', [])
    html_parts   = []

    # ── Two-column grid ────────────────────────────────────────────────────
    html_parts.append('<div class="sf-results-grid">')

    # Left: Outfit Breakdown + Stylist Notes
    html_parts.append(format_analysis_html(result))

    # Right: Product grid
    html_parts.append('<div>')
    if products:
        real_count = sum(1 for p in products if not p.get('is_demo'))
        sub_html = (
            f'<p class="sf-products-sub">'
            f'{real_count} real Trendyol listing{"s" if real_count != 1 else ""} · click to shop'
            f'</p>'
        ) if real_count > 0 else ''
        html_parts.append(f"""
        <div class="sf-products-header">
            <h2 class="sf-products-title">Found {len(products)} Products</h2>
            {sub_html}
        </div>
        """)
        html_parts.append('<div class="sf-products-grid">')
        for product in products:
            html_parts.append(format_product_card_compact(product))
        html_parts.append('</div>')
    else:
        html_parts.append(
            '<div class="sf-status sf-status-info">No matching products found. '
            'Try uploading a different image.</div>'
        )
    html_parts.append('</div>')  # right column
    html_parts.append('</div>')  # sf-results-grid

    return ''.join(html_parts)


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_fashion_image(image: Optional[Image.Image]) -> Tuple[str, str]:
    """
    Analyze fashion image and find products.

    Returns:
        (status_message, results_html)
    """
    if image is None:
        return "Please upload an image first.", ""

    try:
        result = analyzer.analyze_and_find_products(image, max_results=MAX_RESULTS_DISPLAY)

        if result.get('success'):
            products_count = len(result.get('products', []))
            items_count    = len(result.get('fashion_analysis', {}).get('items', []))
            real_count     = sum(1 for p in result.get('products', []) if not p.get('is_demo'))
            status = (
                f"Analysis complete! Detected {items_count} item(s). "
                f"Found {products_count} products ({real_count} real Trendyol listings)."
            )
        else:
            status = f"Analysis issue: {result.get('error', 'Unknown error')}"

        return status, format_results_html(result)

    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(f"Error in analyze_fashion_image: {e}")
        return error_msg, f'<div class="status-error">{error_msg}</div>'


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------

def create_interface():
    """Create and configure the Gradio interface."""

    with gr.Blocks(title="Style Finder AI - Fashion Analysis") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 24px 24px; box-shadow: 0 10px 30px rgba(102,126,234,0.3); position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); animation: sf-rotate 20s linear infinite; pointer-events: none;"></div>
            <div style="position: relative; z-index: 1;">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2); letter-spacing: -0.5px;">Style Finder AI</h1>
                <p style="margin: 0.75rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 1.1rem; font-weight: 500;">AI-Powered Fashion Discovery • Find Matching Pieces on Trendyol</p>
            </div>
            <!-- Dark/light toggle — fully inline onclick so Gradio script sandboxing can't break it -->
            <button id="sf-theme-toggle" title="Toggle dark/light mode"
                onclick="var h=document.documentElement,d=h.classList.contains('dark');d?(h.classList.remove('dark'),h.removeAttribute('data-theme'),this.textContent='🌙'):(h.classList.add('dark'),h.setAttribute('data-theme','dark'),this.textContent='☀️');try{localStorage.setItem('sf-theme',d?'light':'dark')}catch(e){}"
                style="position:fixed;top:4.5rem;right:1.25rem;background:rgba(102,126,234,.85);border:1.5px solid rgba(255,255,255,.35);color:white;border-radius:50%;width:40px;height:40px;font-size:1.15rem;cursor:pointer;backdrop-filter:blur(6px);z-index:100;display:flex;align-items:center;justify-content:center;transition:background .2s;box-shadow:0 4px 12px rgba(0,0,0,.25);">
                🌙
            </button>
            <style>
                @keyframes sf-rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
            </style>
        </div>
        """)

        # ── Main layout ───────────────────────────────────────────────────
        with gr.Row():

            # Left: upload + button + analysis panel
            with gr.Column(scale=3, min_width=600):
                with gr.Row():
                    with gr.Column(scale=1, min_width=340):
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Fashion Image",
                            height=350,
                            show_label=True,
                        )
                        analyze_btn = gr.Button(
                            "Analyze & Find Products",
                            variant="primary",
                            size="lg",
                            elem_classes=["btn-primary"],
                        )
                        status_output = gr.Markdown(
                            "",
                            elem_classes=["status-minimal"],
                            visible=False,
                        )

                    with gr.Column(scale=2, min_width=480):
                        results_output = gr.HTML(
                            value="""
                            <div class='sf-ready'>
                                <div class='sf-ready-title'>Ready to Analyze</div>
                                <div class='sf-ready-sub'>Upload a fashion image and click Analyze to discover matching products on Trendyol</div>
                            </div>
                            """,
                            elem_classes=["analysis-results"],
                        )

            # Right: How It Works sidebar
            with gr.Column(scale=1, min_width=300):
                with gr.Accordion("How It Works", open=False, elem_classes=["how-it-works-sidebar"]):
                    gr.HTML("""
                    <div class="how-it-works-content" style="padding: 1rem 0;">
                        <div style="margin-bottom: 2rem;">
                            <div class="step-item" style="display:flex;align-items:start;gap:1rem;margin-bottom:1.5rem;">
                                <div class="step-number" style="flex-shrink:0;width:40px;height:40px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem;">1</div>
                                <div>
                                    <h4 class="step-title">Upload Your Image</h4>
                                    <p class="step-description">Upload a clear photo of the outfit you want to find matching pieces for.</p>
                                </div>
                            </div>
                            <div class="step-item" style="display:flex;align-items:start;gap:1rem;margin-bottom:1.5rem;">
                                <div class="step-number" style="flex-shrink:0;width:40px;height:40px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem;">2</div>
                                <div>
                                    <h4 class="step-title">Groq Llama 4 Maverick Analysis</h4>
                                    <p class="step-description">Groq Llama 4 Maverick (17B, 128 experts) analyzes the image and returns structured fashion data: item types, colors, patterns, materials, fit, and gender — in a single API call.</p>
                                    <div class="info-box">
                                        <p class="info-title">VLM: meta-llama/llama-4-maverick-17b-128e-instruct</p>
                                        <p class="info-text">Meta's best vision model — reasons about images with structured JSON output. No multi-step guesswork.</p>
                                        <p class="info-subtitle">Runs via Groq LPU — free tier, no credit card needed</p>
                                    </div>
                                </div>
                            </div>
                            <div class="step-item" style="display:flex;align-items:start;gap:1rem;margin-bottom:1.5rem;">
                                <div class="step-number" style="flex-shrink:0;width:40px;height:40px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem;">3</div>
                                <div>
                                    <h4 class="step-title">Smart Turkish Query Generation</h4>
                                    <p class="step-description">VLM-extracted attributes are translated to Turkish and combined into precise Trendyol search queries (e.g. "Kadın Lacivert Slim Jean").</p>
                                </div>
                            </div>
                            <div class="step-item" style="display:flex;align-items:start;gap:1rem;margin-bottom:1.5rem;">
                                <div class="step-number" style="flex-shrink:0;width:40px;height:40px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem;">4</div>
                                <div>
                                    <h4 class="step-title">Trendyol JSON API Search</h4>
                                    <p class="step-description">Queries Trendyol's internal product search API for real listings — actual prices, images, and product pages. No scraping, no bot detection.</p>
                                    <div class="info-box">
                                        <p class="info-title">Visual Similarity: Fashion-CLIP</p>
                                        <ul class="info-list">
                                            <li>patrickjohncyh/fashion-clip — trained on 800K+ fashion pairs</li>
                                            <li>Text-image cosine similarity ranks product relevance</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                            <div class="step-item" style="display:flex;align-items:start;gap:1rem;">
                                <div class="step-number" style="flex-shrink:0;width:40px;height:40px;background:linear-gradient(135deg,#667eea,#764ba2);border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:700;font-size:1.1rem;">5</div>
                                <div>
                                    <h4 class="step-title">Shop on Trendyol</h4>
                                    <p class="step-description">Click any product card to view the real listing on Trendyol.com with accurate price, brand, and availability.</p>
                                </div>
                            </div>
                        </div>
                        <div class="tips-section">
                            <h4 class="step-title">Tips for Best Results</h4>
                            <ul class="tips-list">
                                <li>Use clear, well-lit images</li>
                                <li>Ensure clothing items are visible</li>
                                <li>Full-body or upper-body photos work best</li>
                                <li>Single outfit per image for accuracy</li>
                            </ul>
                        </div>
                    </div>
                    """)

        # ── Event handlers ────────────────────────────────────────────────
        LOADING_HTML = """
        <div class="sf-loading">
            <div class="sf-spinner"></div>
            <div class="sf-loading-text">Analyzing your outfit...</div>
            <div class="sf-loading-sub">Groq Llama 4 Maverick is identifying clothing items, colors, and style</div>
        </div>
        """

        def show_analyzing():
            return gr.update(value=LOADING_HTML)

        def hide_status():
            return gr.update(value="", visible=False)

        analyze_btn.click(
            fn=show_analyzing,
            inputs=None,
            outputs=[results_output],
        ).then(
            fn=analyze_fashion_image,
            inputs=[image_input],
            outputs=[status_output, results_output],
        )

        image_input.upload(
            fn=hide_status,
            inputs=None,
            outputs=status_output,
        )

    return demo


if __name__ == "__main__":
    print("Starting Style Finder AI...")
    print("Loading models (this may take a moment on first run)...")

    demo = create_interface()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css,
    )

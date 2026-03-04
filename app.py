"""
Style Finder AI - Main Application
Computer Vision-Based Fashion Analysis with Trendyol Integration
"""

import gradio as gr
from PIL import Image
import sys
import os
from typing import Tuple, Optional
import time

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


def format_product_card(product: dict) -> str:
    """
    Format a product as an HTML card
    
    Args:
        product: Product dictionary
        
    Returns:
        HTML string for product card
    """
    name = product.get('name', 'Unknown Product')
    price_text = product.get('price_text', 'Fiyat bilgisi yok')
    image_url = product.get('image_url', '')
    product_url = product.get('url', '#')
    brand = product.get('brand', '')
    similarity_score = product.get('similarity_score', 0.0)
    
    # Format similarity as percentage
    similarity_percent = int(similarity_score * 100)
    
    # Build HTML card
    card_html = f"""
    <div class="product-card fade-in">
        <img src="{image_url}" alt="{name}" class="product-image" onerror="this.src='https://via.placeholder.com/300x300?text=No+Image'">
        <div class="product-info">
            <div class="product-name">{name}</div>
            {f'<div class="product-brand">Brand: {brand}</div>' if brand else ''}
            <div class="product-price">{price_text}</div>
            <div class="product-similarity">{similarity_percent}% Match</div>
            <a href="{product_url}" target="_blank" class="product-link">View on Trendyol →</a>
        </div>
    </div>
    """
    
    return card_html


def format_results_html(result: dict) -> str:
    """
    Format analysis results as HTML - responsive CSS-class-driven design.

    Args:
        result: Analysis result dictionary

    Returns:
        Formatted HTML string
    """
    if not result.get('success'):
        error = result.get('error', 'Unknown error occurred')
        return f"""
        <div class="sf-error-panel">
            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: #dc2626;">
                Unable to Analyze Image
            </div>
            <div style="font-size: 1rem; line-height: 1.6;">
                {error}
            </div>
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
    products = result.get('products', [])

    html_parts = []

    # Responsive two-column layout (CSS handles breakpoints)
    html_parts.append('<div class="sf-results-grid">')

    # ── Left: Analysis panel ───────────────────────────────────────────────
    html_parts.append('<div class="sf-analysis-panel">')
    html_parts.append('<h3 class="sf-analysis-title">Outfit Breakdown</h3>')

    if fashion_data.get('items'):
        for item in fashion_data['items']:
            item_type = item.get("type", "Unknown").title()
            raw_color = item.get("color", "")
            item_color = raw_color.title() if raw_color not in ("unknown", "", None) else ""
            item_display = f"{item_color} {item_type}".strip() if item_color else item_type

            extra_parts = []
            pattern = item.get("pattern", "")
            material = item.get("material", "")
            fit = item.get("fit", "")
            description = item.get("description", "")
            if pattern and pattern not in ("solid", "unknown", ""):
                extra_parts.append(pattern.title())
            if material and material not in ("unknown", ""):
                extra_parts.append(material.title())
            if fit and fit not in ("unknown", ""):
                extra_parts.append(fit.title())

            dot_html = _color_dot(raw_color) if raw_color else ""
            tags_html = "".join(
                f'<span class="sf-chip-tag">{p}</span>' for p in extra_parts
            )

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

    style_tips = fashion_data.get('stylist_notes', [])
    if style_tips:
        html_parts.append('<div class="sf-tips-section">')
        html_parts.append('<h4 class="sf-tips-title">Stylist Notes</h4>')
        for tip in style_tips:
            if tip and tip.strip():
                html_parts.append(f'<div class="sf-tip-item">{tip}</div>')
        html_parts.append('</div>')  # sf-tips-section

    html_parts.append('</div>')  # sf-analysis-panel

    # ── Right: Product grid ────────────────────────────────────────────────
    html_parts.append('<div>')
    if products:
        real_count = sum(1 for p in products if not p.get('is_demo'))
        if real_count > 0:
            sub_text = f"{real_count} real Trendyol listing{'s' if real_count != 1 else ''} · click to shop"
        else:
            sub_text = "Search links · Trendyol may be temporarily unreachable from this network"
        html_parts.append(f"""
        <div class="sf-products-header">
            <h2 class="sf-products-title">Found {len(products)} Products</h2>
            <p class="sf-products-sub">{sub_text}</p>
        </div>
        """)
        html_parts.append('<div class="sf-products-grid">')
        for product in products:
            html_parts.append(format_product_card_compact(product))
        html_parts.append('</div>')  # sf-products-grid
    else:
        html_parts.append('<div class="sf-status sf-status-info">No matching products found. Try uploading a different image.</div>')

    html_parts.append('</div>')  # right column
    html_parts.append('</div>')  # sf-results-grid

    return ''.join(html_parts)


def format_product_card_compact(product: dict) -> str:
    """Format a product as a compact HTML card using CSS classes."""
    name = product.get('name', 'Unknown Product')
    product_url = product.get('url', '#')
    image_url = product.get('image_url', '')
    brand = product.get('brand', '')
    price_text = product.get('price_text', '')
    similarity_score = product.get('similarity_score', 0.0)
    similarity_percent = int(similarity_score * 100)
    is_demo = product.get('is_demo', False)

    # Clean display name
    display_name = name.replace(" — Trendyol'da Ara", "").replace(" - Trendyol'da Ara", "").strip()
    if len(display_name) > 60:
        display_name = display_name[:57] + "..."

    # Image section
    if image_url:
        img_html = f'<img src="{image_url}" alt="{display_name}" class="sf-card-img" onerror="this.style.display=\'none\'">'
    else:
        img_html = ''

    badge_class = "sf-badge sf-badge-real" if not is_demo else "sf-badge sf-badge-demo"
    badge_label = f"{similarity_percent}% Match" if not is_demo else "Search Link"

    brand_html = f'<div class="sf-card-brand">{brand}</div>' if brand else ''
    price_html = f'<div class="sf-card-price">{price_text}</div>' if price_text else ''

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


def analyze_fashion_image(image: Optional[Image.Image]) -> Tuple[str, str]:
    """
    Main function to analyze fashion image and find products
    
    Args:
        image: PIL Image object from Gradio
        
    Returns:
        Tuple of (status_message, results_html)
    """
    if image is None:
        return "Please upload an image first.", ""
    
    try:
        # Update status
        status = "Analyzing fashion image... This may take a minute."
        
        # Run analysis
        result = analyzer.analyze_and_find_products(image, max_results=MAX_RESULTS_DISPLAY)
        
        if result.get('success'):
            products_count = len(result.get('products', []))
            items_count = len(result.get('fashion_analysis', {}).get('items', []))
            real_count = sum(1 for p in result.get('products', []) if not p.get('is_demo'))
            status = (
                f"Analysis complete! Detected {items_count} item(s). "
                f"Found {products_count} products ({real_count} real Trendyol listings)."
            )
        else:
            status = f"Analysis issue: {result.get('error', 'Unknown error')}"
        
        # Format results
        results_html = format_results_html(result)
        
        return status, results_html
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(f"Error in analyze_fashion_image: {e}")
        return error_msg, f'<div class="status-error">{error_msg}</div>'


def create_interface():
    """Create and configure the Gradio interface - Professional single-view design"""
    
    with gr.Blocks(title="Style Finder AI - Fashion Analysis") as demo:
        
        # Professional Header with Animation
        gr.HTML("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 24px 24px; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); animation: rotate 20s linear infinite;"></div>
            <div style="position: relative; z-index: 1;">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2); letter-spacing: -0.5px;">Style Finder AI</h1>
                <p style="margin: 0.75rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 1.1rem; font-weight: 500;">AI-Powered Fashion Discovery • Find Matching Pieces on Trendyol</p>
            </div>
            <button id="sf-theme-toggle" onclick="sfToggleTheme()" title="Toggle dark/light mode"
                style="position: absolute; top: 1rem; right: 1.25rem; background: rgba(255,255,255,0.18); border: 1.5px solid rgba(255,255,255,0.35); color: white; border-radius: 50%; width: 40px; height: 40px; font-size: 1.1rem; cursor: pointer; backdrop-filter: blur(6px); z-index: 10; display: flex; align-items: center; justify-content: center; transition: background .2s;">
                🌙
            </button>
            <style>
                @keyframes rotate {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            </style>
            <script>
                function sfToggleTheme() {
                    var html = document.documentElement;
                    var isDark = html.getAttribute('data-theme') === 'dark';
                    html.setAttribute('data-theme', isDark ? '' : 'dark');
                    var btn = document.getElementById('sf-theme-toggle');
                    if (btn) btn.textContent = isDark ? '🌙' : '☀️';
                    try { localStorage.setItem('sf-theme', isDark ? 'light' : 'dark'); } catch(e) {}
                }
                (function() {
                    try {
                        var saved = localStorage.getItem('sf-theme');
                        if (saved === 'dark') {
                            document.documentElement.setAttribute('data-theme', 'dark');
                            var btn = document.getElementById('sf-theme-toggle');
                            if (btn) btn.textContent = '☀️';
                        }
                    } catch(e) {}
                })();
            </script>
        </div>
        """)
        
        # Main Content - Layout with Sidebar
        with gr.Row():
            with gr.Column(scale=3, min_width=600):
                # Main Content Area
                with gr.Row():
                    with gr.Column(scale=1, min_width=350):
                        # Image Upload Section - Compact
                        image_input = gr.Image(
                            type="pil",
                            label="Upload Fashion Image",
                            height=350,
                            show_label=True
                        )
                        
                        # Analyze Button
                        analyze_btn = gr.Button(
                            "Analyze & Find Products",
                            variant="primary",
                            size="lg",
                            elem_classes=["btn-primary"]
                        )
                        
                        # Minimal Status (hidden by default)
                        status_output = gr.Markdown(
                            "",
                            elem_classes=["status-minimal"],
                            visible=False
                        )
                    
                    with gr.Column(scale=2, min_width=500):
                        # Results Section - Immediately Visible
                        results_output = gr.HTML(
                            value="""
                            <div class='sf-ready'>
                                <div class='sf-ready-title'>Ready to Analyze</div>
                                <div class='sf-ready-sub'>Upload a fashion image and click Analyze to discover matching products on Trendyol</div>
                            </div>
                            """,
                            elem_classes=["analysis-results"]
                        )
            
            with gr.Column(scale=1, min_width=300):
                # How It Works Sidebar
                with gr.Accordion("How It Works", open=False, elem_classes=["how-it-works-sidebar"]):
                    gr.HTML("""
                    <div class="how-it-works-content" style="padding: 1rem 0;">
                        <div style="margin-bottom: 2rem;">
                            <div class="step-item" style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div class="step-number" style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">1</div>
                                <div>
                                    <h4 class="step-title">Upload Your Image</h4>
                                    <p class="step-description">Upload a clear photo of the outfit you want to find matching pieces for.</p>
                                </div>
                            </div>

                            <div class="step-item" style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div class="step-number" style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">2</div>
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

                            <div class="step-item" style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div class="step-number" style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">3</div>
                                <div>
                                    <h4 class="step-title">Smart Turkish Query Generation</h4>
                                    <p class="step-description">VLM-extracted attributes are translated to Turkish and combined into precise Trendyol search queries (e.g. "Kadın Lacivert Slim Jean").</p>
                                </div>
                            </div>

                            <div class="step-item" style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div class="step-number" style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">4</div>
                                <div>
                                    <h4 class="step-title">Trendyol JSON API Search</h4>
                                    <p class="step-description">Queries Trendyol's internal product search API for real listings — actual prices, images, and product pages. No scraping, no bot detection.</p>
                                    <div class="info-box">
                                        <p class="info-title">Visual Similarity: Fashion-CLIP</p>
                                        <ul class="info-list">
                                            <li>patrickjohncyh/fashion-clip — trained on 800K+ fashion pairs</li>
                                            <li>Text-image cosine similarity ranks product relevance</li>
                                            <li>No product thumbnail downloads required</li>
                                        </ul>
                                    </div>
                                </div>
                            </div>

                            <div class="step-item" style="display: flex; align-items: start; gap: 1rem;">
                                <div class="step-number" style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">5</div>
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
        
        # Event Handlers
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
            outputs=results_output
        ).then(
            fn=analyze_fashion_image,
            inputs=[image_input],
            outputs=[status_output, results_output]
        )

        image_input.upload(
            fn=hide_status,
            inputs=None,
            outputs=status_output
        )
    
    return demo


if __name__ == "__main__":
    print("Starting Style Finder AI...")
    print("Loading models (this may take a moment on first run)...")
    
    demo = create_interface()
    
    # Launch with sharing enabled for easy access
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css
    )


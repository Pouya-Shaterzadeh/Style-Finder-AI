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


def get_style_tips(overall_style: str, items: list) -> list:
    """
    Generate style tips specific to the detected outfit and items
    
    Args:
        overall_style: Overall style detected (e.g., 'casual', 'formal')
        items: List of detected clothing items with colors and types
        
    Returns:
        List of style tip strings specific to the outfit
    """
    tips = []
    style_lower = overall_style.lower() if overall_style else ''
    
    # Get specific item details
    item_details = {}
    for item in items:
        item_type = item.get('type', '').lower()
        item_color = item.get('color', '').lower()
        if item_type:
            if item_type not in item_details:
                item_details[item_type] = []
            item_details[item_type].append(item_color)
    
    # Generate outfit-specific tips based on detected items
    if items:
        tips.append("<strong>Outfit-Specific Tips:</strong>")
        
        # Tips based on specific item combinations
        has_top = any(t in item_details for t in ['sweater', 'shirt', 't-shirt', 'blouse', 'hoodie'])
        has_bottom = any(t in item_details for t in ['pants', 'jeans', 'shorts', 'skirt'])
        
        if has_top and has_bottom:
            top_item = next((t for t in ['sweater', 'shirt', 't-shirt', 'blouse', 'hoodie'] if t in item_details), None)
            bottom_item = next((t for t in ['pants', 'jeans', 'shorts', 'skirt'] if t in item_details), None)
            
            if top_item and bottom_item:
                top_colors = item_details[top_item]
                bottom_colors = item_details[bottom_item]
                
                # Specific color combination tips
                if 'black' in top_colors and 'blue' in bottom_colors:
                    tips.append(f"• Your {top_item} and {bottom_item} create a classic contrast - add a neutral accessory")
                elif 'blue' in top_colors and 'black' in bottom_colors:
                    tips.append(f"• Your {top_item} and {bottom_item} create a classic contrast - add a neutral accessory")
                elif any(c in ['black', 'white', 'gray'] for c in top_colors) and any(c in ['black', 'white', 'gray'] for c in bottom_colors):
                    tips.append(f"• Monochromatic look - add a pop of color with accessories or shoes")
                else:
                    tips.append(f"• Your {top_item} and {bottom_item} combination works well - consider adding a complementary accessory")
        
        # Specific item tips based on what was detected
        for item_type, colors in item_details.items():
            color_str = colors[0] if colors and colors[0] != 'unknown' else ''
            
            if item_type == 'sweater':
                if 'blue' in colors:
                    tips.append(f"• Your blue sweater pairs well with neutral bottoms - try beige or white")
                elif 'black' in colors:
                    tips.append(f"• Your black sweater is versatile - layer with a white shirt underneath for a preppy look")
                else:
                    tips.append(f"• Sweaters are great for layering - add a collared shirt underneath for texture")
            
            elif item_type in ['pants', 'jeans']:
                if 'blue' in colors:
                    tips.append(f"• Blue {item_type} are a wardrobe staple - pair with any neutral top")
                elif 'black' in colors:
                    tips.append(f"• Black {item_type} elongate the silhouette - perfect for a polished look")
                else:
                    tips.append(f"• Your {color_str} {item_type} can be dressed up or down depending on footwear")
            
            elif item_type in ['shirt', 't-shirt', 'blouse']:
                if 'white' in colors or 'black' in colors:
                    tips.append(f"• Your {color_str} {item_type} is a versatile piece - works with any bottom")
                else:
                    tips.append(f"• Your {color_str} {item_type} adds personality - keep accessories minimal to let it shine")
    
    # Style-specific tips (only if relevant)
    if 'casual' in style_lower and items:
        tips.append("")
        tips.append("<strong>Casual Styling:</strong>")
        tips.append("• Complete the look with comfortable sneakers or flats")
        tips.append("• Add a denim or leather jacket for layering versatility")
    
    elif 'formal' in style_lower or 'elegant' in style_lower:
        tips.append("")
        tips.append("<strong>Formal Styling:</strong>")
        tips.append("• Ensure proper fit - tailored pieces elevate the outfit")
        tips.append("• Add a statement accessory like a watch or belt")
    
    # Limit to 5 most relevant tips
    return tips[:5]


def format_results_html(result: dict) -> str:
    """
    Format analysis results as HTML - Compact, single-view design
    
    Args:
        result: Analysis result dictionary
        
    Returns:
        Formatted HTML string
    """
    if not result.get('success'):
        error = result.get('error', 'Unknown error occurred')
        return f"""
        <div class="status-error" style="padding: 2rem; border-radius: 12px; background: #fef2f2; border: 2px solid #fecaca; color: #991b1b;">
            <div style="font-size: 1.25rem; font-weight: 700; margin-bottom: 1rem; color: #dc2626;">
                Unable to Analyze Image
            </div>
            <div style="font-size: 1rem; line-height: 1.6; color: #7f1d1d;">
                {error}
            </div>
            <div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid #fecaca; font-size: 0.95rem; color: #991b1b;">
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
    
    # Beautiful Layout: Side-by-side Analysis and Products (responsive)
    html_parts.append('<div class="results-grid-container" style="display: grid; grid-template-columns: 340px 1fr; gap: 2rem; margin-top: 1rem;">')
    
    # Left: Professional Fashion Analysis Panel
    html_parts.append('<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 1.5rem; color: white; height: fit-content; position: sticky; top: 1rem; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); border: 1px solid rgba(255,255,255,0.1);">')
    html_parts.append('<div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.2);">')
    html_parts.append('<h3 style="margin: 0; font-size: 1.35rem; font-weight: 700; letter-spacing: -0.3px;">Fashion Analysis</h3>')
    html_parts.append('</div>')
    
    # Detected Items - Professional List (No Emojis)
    if fashion_data.get('items'):
        html_parts.append('<div style="margin-bottom: 1.25rem;">')
        for item in fashion_data['items']:
            item_type = item.get("type", "Unknown").title()
            item_color = item.get("color", "unknown").title() if item.get("color") != "unknown" else ""
            item_display = f"{item_color} {item_type}".strip() if item_color else item_type
            html_parts.append(f'''
            <div style="padding: 0.75rem 1rem; background: rgba(255,255,255,0.2); backdrop-filter: blur(10px); border-radius: 12px; margin-bottom: 0.75rem; font-size: 0.95rem; font-weight: 500; transition: all 0.2s ease; border: 1px solid rgba(255,255,255,0.1);">
                <span>{item_display}</span>
            </div>
            ''')
        html_parts.append('</div>')
    
    # Add Style Tips Section (removed style/gender badges as requested)
    style_tips = get_style_tips(fashion_data.get('overall_style', ''), fashion_data.get('items', []))
    if style_tips:
        html_parts.append('<div style="margin-top: 1.5rem; padding-top: 1.5rem; border-top: 1px solid rgba(255,255,255,0.2);">')
        html_parts.append('<h4 style="margin: 0 0 1rem 0; font-size: 1.1rem; font-weight: 700; letter-spacing: -0.2px;">Style Tips</h4>')
        html_parts.append('<div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 12px; padding: 1rem;">')
        for tip in style_tips[:5]:  # Limit to 5 most relevant tips
            if tip.strip():  # Skip empty lines
                html_parts.append(f'<div style="margin-bottom: 0.75rem; font-size: 0.9rem; line-height: 1.5; opacity: 0.95;">{tip}</div>')
        html_parts.append('</div>')
        html_parts.append('</div>')
    
    html_parts.append('</div>')  # End left column
    
    # Right: Products Grid - Beautiful & Immediately Visible
    html_parts.append('<div>')
    if products:
        html_parts.append(f'''
        <div style="margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 2px solid #e2e8f0;">
            <h2 style="margin: 0 0 0.5rem 0; font-size: 1.75rem; font-weight: 800; color: #1e293b; letter-spacing: -0.5px;">Found {len(products)} Products</h2>
            <p style="margin: 0; color: #64748b; font-size: 0.95rem;">Click any product to view on Trendyol</p>
        </div>
        ''')
        html_parts.append('<div class="products-grid-compact">')
        
        for product in products:
            html_parts.append(format_product_card_compact(product))
        
        html_parts.append('</div>')
    else:
        html_parts.append('<div class="status-info">No matching products found. Try uploading a different image.</div>')
    
    html_parts.append('</div>')  # End right column
    html_parts.append('</div>')  # End grid
    
    return ''.join(html_parts)


def format_product_card_compact(product: dict) -> str:
    """
    Format a product as a compact HTML card for single-view display
    
    Args:
        product: Product dictionary
        
    Returns:
        HTML string for compact product card
    """
    name = product.get('name', 'Unknown Product')
    product_url = product.get('url', '#')
    similarity_score = product.get('similarity_score', 0.0)
    similarity_percent = int(similarity_score * 100)
    
    # Extract item name from product name (remove "Trendyol'da Ara" suffix)
    display_name = name.replace(" - Trendyol'da Ara", "").strip()
    
    # Build compact card (No Emojis) - Mobile-friendly with visible link
    card_html = f"""
    <div class="product-card-compact">
        <a href="{product_url}" target="_blank" rel="noopener noreferrer" class="product-card-link">
            <div class="product-card-content">
                <div class="product-details-compact">
                    <div class="product-name-compact">{display_name}</div>
                    <div class="product-match-badge">{similarity_percent}% Match</div>
                    <div class="product-url-mobile">View on Trendyol →</div>
                </div>
                <div class="product-arrow">→</div>
            </div>
        </a>
    </div>
    """
    
    return card_html


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
            status = f"Analysis complete! Found {products_count} matching products."
        else:
            status = f"Analysis completed with issues: {result.get('error', 'Unknown error')}"
        
        # Format results
        results_html = format_results_html(result)
        
        return status, results_html
        
    except Exception as e:
        error_msg = f"An error occurred: {str(e)}"
        print(f"Error in analyze_fashion_image: {e}")
        return error_msg, f'<div class="status-error">{error_msg}</div>'


def create_interface():
    """Create and configure the Gradio interface - Professional single-view design"""
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="Style Finder AI - Fashion Analysis",
        css=custom_css
    ) as demo:
        
        # Professional Header with Animation
        gr.HTML("""
        <div style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); margin: -1rem -1rem 2rem -1rem; border-radius: 0 0 24px 24px; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3); position: relative; overflow: hidden;">
            <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%); animation: rotate 20s linear infinite;"></div>
            <div style="position: relative; z-index: 1;">
                <h1 style="margin: 0; font-size: 2.5rem; font-weight: 800; color: white; text-shadow: 0 2px 10px rgba(0,0,0,0.2); letter-spacing: -0.5px;">Style Finder AI</h1>
                <p style="margin: 0.75rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 1.1rem; font-weight: 500;">AI-Powered Fashion Discovery • Find Matching Pieces on Trendyol</p>
            </div>
            <style>
                @keyframes rotate {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            </style>
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
                            <div style='text-align: center; padding: 5rem 2rem; color: #64748b; background: linear-gradient(135deg, #f5f7fa 0%, #ffffff 100%); border-radius: 20px; border: 2px dashed #e2e8f0;'>
                                <div style='font-size: 1.25rem; font-weight: 600; color: #475569; margin-bottom: 0.5rem;'>Ready to Analyze</div>
                                <div style='font-size: 1rem; color: #94a3b8;'>Upload a fashion image and click analyze to discover matching products</div>
                            </div>
                            """,
                            elem_classes=["analysis-results"]
                        )
            
            with gr.Column(scale=1, min_width=300):
                # How It Works Sidebar
                with gr.Accordion("How It Works", open=False, elem_classes=["how-it-works-sidebar"]):
                    gr.HTML("""
                    <div style="padding: 1rem 0;">
                        <div style="margin-bottom: 2rem;">
                            <div style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">1</div>
                                <div>
                                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; color: #1e293b;">Upload Your Image</h4>
                                    <p style="margin: 0; font-size: 0.9rem; color: #64748b; line-height: 1.5;">Upload a clear photo of the outfit you want to find matching pieces for.</p>
                                </div>
                            </div>
                            
                            <div style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">2</div>
                                <div>
                                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; color: #1e293b;">AI Agent Analysis</h4>
                                    <p style="margin: 0; font-size: 0.9rem; color: #64748b; line-height: 1.5;">Our Vision Language Model (VLM) is instructed to act as a specialized <strong>Fashion & Style Analyzer AI Agent</strong>. It analyzes the image to extract clothing items, colors, patterns, styles, and gender with precision.</p>
                                    <div style="margin-top: 0.75rem; padding: 0.75rem; background: rgba(102, 126, 234, 0.1); border-radius: 8px; border-left: 3px solid #667eea;">
                                        <p style="margin: 0 0 0.5rem 0; font-size: 0.85rem; color: #475569; font-weight: 700;">VLM Model: BLIP (Salesforce)</p>
                                        <p style="margin: 0 0 0.5rem 0; font-size: 0.8rem; color: #64748b; line-height: 1.4;">The model is prompted with specialized instructions to play the role of a Fashion Analyzer agent, extracting structured fashion data from images.</p>
                                        <p style="margin: 0; font-size: 0.75rem; color: #94a3b8; font-style: italic;">AI Agent Role: Fashion/Style Analyzer</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div style="display: flex; align-items: start; gap: 1rem; margin-bottom: 1.5rem;">
                                <div style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">3</div>
                                <div>
                                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; color: #1e293b;">Smart Matching</h4>
                                    <p style="margin: 0; font-size: 0.9rem; color: #64748b; line-height: 1.5;">The AI agent's extracted fashion attributes are translated to Turkish and used to search Trendyol for precise product matches.</p>
                                </div>
                            </div>
                            
                            <div style="display: flex; align-items: start; gap: 1rem;">
                                <div style="flex-shrink: 0; width: 40px; height: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; font-size: 1.1rem;">4</div>
                                <div>
                                    <h4 style="margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; color: #1e293b;">Shop on Trendyol</h4>
                                    <p style="margin: 0; font-size: 0.9rem; color: #64748b; line-height: 1.5;">Click any product link to view matching items directly on Trendyol.com.</p>
                                </div>
                            </div>
                        </div>
                        
                        <div style="padding-top: 1.5rem; border-top: 1px solid #e2e8f0; margin-top: 1.5rem;">
                            <h4 style="margin: 0 0 0.75rem 0; font-size: 0.95rem; font-weight: 700; color: #1e293b;">Tips for Best Results</h4>
                            <ul style="margin: 0; padding-left: 1.25rem; font-size: 0.85rem; color: #64748b; line-height: 1.8;">
                                <li>Use clear, well-lit images</li>
                                <li>Ensure clothing items are visible</li>
                                <li>Full-body or upper-body photos work best</li>
                                <li>Single outfit per image for accuracy</li>
                            </ul>
                        </div>
                    </div>
                    """)
        
        # Event Handlers
        def show_analyzing():
            return gr.update(value="Analyzing...", visible=True)
        
        def hide_status():
            return gr.update(value="", visible=False)
        
        analyze_btn.click(
            fn=show_analyzing,
            inputs=None,
            outputs=status_output
        ).then(
            fn=analyze_fashion_image,
            inputs=[image_input],
            outputs=[status_output, results_output]
        ).then(
            fn=hide_status,
            inputs=None,
            outputs=status_output
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
        show_error=True
    )


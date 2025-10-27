import numpy as np
from PIL import Image, ImageDraw
import io
import base64

def create_heatmap_overlay(heatmap, original_image):
    """Create heatmap overlay visualization"""
    # Convert heatmap to PIL Image
    heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(original_image.size)
    
    # Create colored heatmap
    heatmap_colored = Image.new('RGB', original_image.size)
    # Add your heatmap coloring logic here
    
    return heatmap_colored
"""
八叉树量化器
"""
import numpy as np
from PIL import Image
from .base import BaseQuantizer

class OctreeQuantizer(BaseQuantizer):
    """八叉树量化器 (基于 Pillow Fast Octree)"""
    
    def quantize(self, image: np.ndarray, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
        if image is None:
            return None, None
            
        pil_image = Image.fromarray(image)
        
        # method=2: Fast Octree
        q_image = pil_image.quantize(colors=n_colors, method=2)
        
        palette = q_image.getpalette()
        if palette:
            palette = np.array(palette[:n_colors*3]).reshape(-1, 3).astype(np.uint8)
        else:
            palette = np.zeros((n_colors, 3), dtype=np.uint8)
            
        quantized_image = np.array(q_image.convert('RGB'))
        
        return quantized_image, palette

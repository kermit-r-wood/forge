"""
中值切割量化器
"""
import numpy as np
from PIL import Image
from .base import BaseQuantizer

class MedianCutQuantizer(BaseQuantizer):
    """中值切割量化器 (基于 Pillow)"""
    
    def quantize(self, image: np.ndarray, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
        if image is None:
            return None, None
            
        # 转换为 Pillow Image
        if image.ndim == 3 and image.shape[2] == 3:
            # OpenCV 是 BGR, Pillow 是 RGB
            # 假设输入 image 已经是 RGB (由 Analyzer 统一处理)
            # 或在这里处理 BGR->RGB?
            # 这里的 BaseFilter 默认处理 RGB 或 BGR? OpenCV 通常用 BGR
            # 我们约定 core 内部处理尽量统一，假设输入是 RGB
            pil_image = Image.fromarray(image)
        else:
            return image, None
            
        # 使用 Pillow 的 quantize (实现是 Median Cut 或 Octree 变种)
        # method=0 是 median cut, method=1 是 max coverage, method=2 是 fast octree
        # Pillow 默认 quantize 使用 Median Cut (method=0)
        q_image = pil_image.quantize(colors=n_colors, method=0)
        
        # 获取调色板
        palette = q_image.getpalette()
        if palette:
            # Pillow palette 是 [r, g, b, r, g, b, ...] 平铺
            palette = np.array(palette[:n_colors*3]).reshape(-1, 3).astype(np.uint8)
        else:
            palette = np.zeros((n_colors, 3), dtype=np.uint8)
            
        # 转换回 numpy
        # q_image 是 P模式 (索引)，需要转回 RGB 查看效果，或者保留索引
        # 这里返回 RGB 图像以便后续处理
        quantized_image = np.array(q_image.convert('RGB'))
        
        return quantized_image, palette

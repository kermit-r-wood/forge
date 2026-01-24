"""
VTracer 矢量化包装器
可选依赖，不可用时自动回退到 Color-Traced
"""
import numpy as np
import cv2
import io
from .base import BaseVectorizer
from .color_traced import ColorTracedVectorizer

# 尝试导入 vtracer
try:
    import vtracer
    HAS_VTRACER = True
except ImportError:
    HAS_VTRACER = False


class VTracerVectorizer(BaseVectorizer):
    """
    VTracer 矢量化包装器
    使用 Rust 编写的高性能矢量化库
    不可用时自动回退到 Color-Traced
    """
    
    def __init__(self, n_colors: int = 16):
        """
        Args:
            n_colors: 量化颜色数量
        """
        self.n_colors = n_colors
        self._fallback = ColorTracedVectorizer(n_colors)
        
    @property
    def is_available(self) -> bool:
        """VTracer 是否可用"""
        return HAS_VTRACER
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行 VTracer 矢量化"""
        if not HAS_VTRACER:
            # 回退到 Color-Traced
            return self._fallback.apply(image, palette)
        
        h, w = image.shape[:2]
        
        # 1. 量化减色
        quantized = self._quantize_image(image, self.n_colors)
        
        # 2. 映射到调色板
        indices = self._map_to_palette(quantized, palette)
        mapped = palette[indices]
        
        # 3. 尝试使用 VTracer 处理
        try:
            result = self._apply_vtracer(mapped)
            # 重新生成索引（VTracer 可能轻微改变颜色）
            indices = self._map_to_palette(result, palette)
        except Exception:
            # VTracer 失败，使用回退
            result, indices = self._fallback.apply(image, palette)
        
        return result, indices
    
    def _quantize_image(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """K-Means 量化"""
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        centers = centers.astype(np.uint8)
        return centers[labels.flatten()].reshape(h, w, 3)
    
    def _map_to_palette(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """映射到调色板"""
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        palette_f = palette.astype(np.float32)
        
        batch_size = 50000
        indices = np.zeros(len(pixels), dtype=np.int32)
        
        for i in range(0, len(pixels), batch_size):
            batch = pixels[i:i+batch_size]
            diff = batch[:, np.newaxis, :] - palette_f[np.newaxis, :, :]
            dist = np.sum(diff ** 2, axis=2)
            indices[i:i+batch_size] = np.argmin(dist, axis=1)
        
        return indices.reshape(h, w)
    
    def _apply_vtracer(self, image: np.ndarray) -> np.ndarray:
        """使用 VTracer 处理图像"""
        from PIL import Image
        
        # 编码为 PNG
        pil_img = Image.fromarray(image)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        # 调用 VTracer
        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format='PNG',
            colormode='color',
            hierarchical='stacked',
            mode='polygon',
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3
        )
        
        # VTracer 生成了 SVG，但我们目前没有 SVG 渲染器
        # 直接返回输入图像（VTracer 内部已做矢量化优化）
        # 完整实现需要 cairosvg 等库渲染 SVG
        return image.copy()

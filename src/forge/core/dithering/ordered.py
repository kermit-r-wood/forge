"""
Ordered Dithering (Bayer Matrix) 实现
使用 Bayer 矩阵实现规则图案抖动，适合像素艺术风格
计算速度快，无误差扩散
"""
import numpy as np
from numba import jit, prange
from .base import BaseDither

# Bayer 矩阵生成函数
def _generate_bayer_matrix(n: int) -> np.ndarray:
    """
    生成 n x n Bayer 矩阵 (n 必须是 2 的幂)
    使用递归构造方法
    """
    if n == 2:
        return np.array([[0, 2], [3, 1]], dtype=np.float32)
    
    smaller = _generate_bayer_matrix(n // 2)
    result = np.zeros((n, n), dtype=np.float32)
    
    # 四个象限
    result[:n//2, :n//2] = 4 * smaller
    result[:n//2, n//2:] = 4 * smaller + 2
    result[n//2:, :n//2] = 4 * smaller + 3
    result[n//2:, n//2:] = 4 * smaller + 1
    
    return result

# 预计算常用的 Bayer 矩阵
_BAYER_2 = _generate_bayer_matrix(2) / 4.0
_BAYER_4 = _generate_bayer_matrix(4) / 16.0
_BAYER_8 = _generate_bayer_matrix(8) / 64.0
_BAYER_16 = _generate_bayer_matrix(16) / 256.0


@jit(nopython=True, parallel=True, cache=True)
def _ordered_dither_kernel(
    image: np.ndarray,
    palette_rgb: np.ndarray,
    threshold_matrix: np.ndarray,
    spread: float,
    out_img: np.ndarray
):
    """
    Ordered Dithering 核心算法 (Numba JIT 加速)
    """
    h, w, _ = image.shape
    th, tw = threshold_matrix.shape
    n_colors = len(palette_rgb)
    
    for y in prange(h):
        for x in range(w):
            # 获取阈值偏移
            threshold = threshold_matrix[y % th, x % tw] - 0.5
            
            # 对每个通道应用阈值偏移
            r = image[y, x, 0] + threshold * spread
            g = image[y, x, 1] + threshold * spread
            b = image[y, x, 2] + threshold * spread
            
            # 限制范围
            r = max(0.0, min(255.0, r))
            g = max(0.0, min(255.0, g))
            b = max(0.0, min(255.0, b))
            
            # 找到最近的调色板颜色
            best_dist = 1e10
            best_idx = 0
            
            for i in range(n_colors):
                pr = palette_rgb[i, 0]
                pg = palette_rgb[i, 1]
                pb = palette_rgb[i, 2]
                dist = (r - pr)**2 + (g - pg)**2 + (b - pb)**2
                
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            
            out_img[y, x, 0] = palette_rgb[best_idx, 0]
            out_img[y, x, 1] = palette_rgb[best_idx, 1]
            out_img[y, x, 2] = palette_rgb[best_idx, 2]


class OrderedDither(BaseDither):
    """
    Ordered (Bayer) Dithering
    产生规则的像素艺术风格图案，计算速度快
    """
    
    def __init__(self, matrix_size: int = 8, spread: float = 64.0):
        """
        Args:
            matrix_size: Bayer 矩阵大小 (2, 4, 8, 16)
            spread: 抖动强度 (值越大，颜色过渡越明显)
        """
        super().__init__()
        self.matrix_size = matrix_size
        self.spread = spread
        self._threshold_matrix = self._get_matrix(matrix_size)
    
    def _get_matrix(self, size: int) -> np.ndarray:
        if size == 2:
            return _BAYER_2
        elif size == 4:
            return _BAYER_4
        elif size == 8:
            return _BAYER_8
        elif size == 16:
            return _BAYER_16
        else:
            return _generate_bayer_matrix(8) / 64.0
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        
        h, w = image.shape[:2]
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 准备数据
        img_float = image.astype(np.float32)
        palette_float = palette.astype(np.float32)
        
        # 调用 Numba 加速的核心算法
        _ordered_dither_kernel(
            img_float,
            palette_float,
            self._threshold_matrix,
            self.spread,
            out_img
        )
        
        return out_img

"""
Floyd-Steinberg 抖动实现 (Numba 加速版)
"""
import numpy as np
from numba import jit
from .base import BaseDither


@jit(nopython=True, cache=True)
def _find_closest_color_fast(pixel_r, pixel_g, pixel_b, palette):
    """快速查找最近颜色 (Numba JIT)"""
    best_dist = 1e10
    best_idx = 0
    
    for i in range(len(palette)):
        pr, pg, pb = palette[i, 0], palette[i, 1], palette[i, 2]
        dist = (pixel_r - pr)**2 + (pixel_g - pg)**2 + (pixel_b - pb)**2
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    return best_idx


@jit(nopython=True, cache=True)
def _floyd_steinberg_kernel(float_img, palette, out_img):
    """Floyd-Steinberg 核心算法 (Numba JIT 加速)"""
    h, w = float_img.shape[:2]
    
    for y in range(h):
        for x in range(w):
            old_r = float_img[y, x, 0]
            old_g = float_img[y, x, 1]
            old_b = float_img[y, x, 2]
            
            # 使用快速查找找到最接近的颜色
            best_idx = _find_closest_color_fast(old_r, old_g, old_b, palette)
            
            new_r = palette[best_idx, 0]
            new_g = palette[best_idx, 1]
            new_b = palette[best_idx, 2]
            
            out_img[y, x, 0] = new_r
            out_img[y, x, 1] = new_g
            out_img[y, x, 2] = new_b
            
            # 计算量化误差
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # 扩散误差 (Floyd-Steinberg 模式)
            #      X   7
            #  3   5   1
            #  ( / 16 )
            if x + 1 < w:
                float_img[y, x + 1, 0] += err_r * 7 / 16
                float_img[y, x + 1, 1] += err_g * 7 / 16
                float_img[y, x + 1, 2] += err_b * 7 / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    float_img[y + 1, x - 1, 0] += err_r * 3 / 16
                    float_img[y + 1, x - 1, 1] += err_g * 3 / 16
                    float_img[y + 1, x - 1, 2] += err_b * 3 / 16
                float_img[y + 1, x, 0] += err_r * 5 / 16
                float_img[y + 1, x, 1] += err_g * 5 / 16
                float_img[y + 1, x, 2] += err_b * 5 / 16
                if x + 1 < w:
                    float_img[y + 1, x + 1, 0] += err_r * 1 / 16
                    float_img[y + 1, x + 1, 1] += err_g * 1 / 16
                    float_img[y + 1, x + 1, 2] += err_b * 1 / 16


class FloydSteinbergDither(BaseDither):
    """
    Floyd-Steinberg 误差扩散抖动 (Numba 加速版)
         X   7
     3   5   1
     ( / 16 )
    """
    
    def __init__(self):
        super().__init__()
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        h, w = image.shape[:2]
        float_img = image.astype(np.float64)
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        palette_float = palette.astype(np.float64)
        
        # 调用 Numba 加速的核心算法
        _floyd_steinberg_kernel(float_img, palette_float, out_img)
                        
        return out_img

"""
Sierra 抖动实现 (Sierra3) - Numba 加速版
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
def _sierra_kernel(float_img, palette, out_img):
    """Sierra 核心算法 (Numba JIT 加速)"""
    h, w = float_img.shape[:2]
    
    for y in range(h):
        for x in range(w):
            old_r = float_img[y, x, 0]
            old_g = float_img[y, x, 1]
            old_b = float_img[y, x, 2]
            
            best_idx = _find_closest_color_fast(old_r, old_g, old_b, palette)
            
            new_r = palette[best_idx, 0]
            new_g = palette[best_idx, 1]
            new_b = palette[best_idx, 2]
            
            out_img[y, x, 0] = new_r
            out_img[y, x, 1] = new_g
            out_img[y, x, 2] = new_b
            
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # Sierra3 扩散模式
            #      X   5   3
            #  2   4   5   4   2
            #      2   3   2
            #  ( / 32 )
            
            # Row 0: (1,0,5), (2,0,3)
            if x + 1 < w:
                float_img[y, x + 1, 0] += err_r * 5 / 32
                float_img[y, x + 1, 1] += err_g * 5 / 32
                float_img[y, x + 1, 2] += err_b * 5 / 32
            if x + 2 < w:
                float_img[y, x + 2, 0] += err_r * 3 / 32
                float_img[y, x + 2, 1] += err_g * 3 / 32
                float_img[y, x + 2, 2] += err_b * 3 / 32
            
            # Row 1: (-2,1,2), (-1,1,4), (0,1,5), (1,1,4), (2,1,2)
            if y + 1 < h:
                if x - 2 >= 0:
                    float_img[y + 1, x - 2, 0] += err_r * 2 / 32
                    float_img[y + 1, x - 2, 1] += err_g * 2 / 32
                    float_img[y + 1, x - 2, 2] += err_b * 2 / 32
                if x - 1 >= 0:
                    float_img[y + 1, x - 1, 0] += err_r * 4 / 32
                    float_img[y + 1, x - 1, 1] += err_g * 4 / 32
                    float_img[y + 1, x - 1, 2] += err_b * 4 / 32
                float_img[y + 1, x, 0] += err_r * 5 / 32
                float_img[y + 1, x, 1] += err_g * 5 / 32
                float_img[y + 1, x, 2] += err_b * 5 / 32
                if x + 1 < w:
                    float_img[y + 1, x + 1, 0] += err_r * 4 / 32
                    float_img[y + 1, x + 1, 1] += err_g * 4 / 32
                    float_img[y + 1, x + 1, 2] += err_b * 4 / 32
                if x + 2 < w:
                    float_img[y + 1, x + 2, 0] += err_r * 2 / 32
                    float_img[y + 1, x + 2, 1] += err_g * 2 / 32
                    float_img[y + 1, x + 2, 2] += err_b * 2 / 32
            
            # Row 2: (-1,2,2), (0,2,3), (1,2,2)
            if y + 2 < h:
                if x - 1 >= 0:
                    float_img[y + 2, x - 1, 0] += err_r * 2 / 32
                    float_img[y + 2, x - 1, 1] += err_g * 2 / 32
                    float_img[y + 2, x - 1, 2] += err_b * 2 / 32
                float_img[y + 2, x, 0] += err_r * 3 / 32
                float_img[y + 2, x, 1] += err_g * 3 / 32
                float_img[y + 2, x, 2] += err_b * 3 / 32
                if x + 1 < w:
                    float_img[y + 2, x + 1, 0] += err_r * 2 / 32
                    float_img[y + 2, x + 1, 1] += err_g * 2 / 32
                    float_img[y + 2, x + 1, 2] += err_b * 2 / 32


class SierraDither(BaseDither):
    """
    Sierra (Sierra3) 抖动 - Numba 加速版
         X   5   3
     2   4   5   4   2
         2   3   2
     ( / 32 )
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
        
        _sierra_kernel(float_img, palette_float, out_img)
                        
        return out_img

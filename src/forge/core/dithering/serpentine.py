"""
蛇形 Floyd-Steinberg 抖动实现 (Numba 加速版) - LAB 色彩空间
交替扫描方向以消除水平条纹
"""
import numpy as np
from numba import jit
from .base import BaseDither, _find_closest_color_lab, precompute_palette_lab


@jit(nopython=True, cache=True)
def _serpentine_fs_kernel_lab(float_img, palette_rgb, palette_lab, out_img):
    """蛇形 Floyd-Steinberg 核心算法 (Numba JIT 加速) - 使用 LAB 色彩匹配"""
    h, w = float_img.shape[:2]
    
    for y in range(h):
        # 蛇形扫描：偶数行从左到右，奇数行从右到左
        if y % 2 == 0:
            x_range_start, x_range_end, x_step = 0, w, 1
        else:
            x_range_start, x_range_end, x_step = w - 1, -1, -1
            
        x = x_range_start
        while (x_step > 0 and x < x_range_end) or (x_step < 0 and x > x_range_end):
            old_r = float_img[y, x, 0]
            old_g = float_img[y, x, 1]
            old_b = float_img[y, x, 2]
            
            best_idx = _find_closest_color_lab(old_r, old_g, old_b, palette_lab)
            
            new_r = palette_rgb[best_idx, 0]
            new_g = palette_rgb[best_idx, 1]
            new_b = palette_rgb[best_idx, 2]
            
            out_img[y, x, 0] = int(new_r)
            out_img[y, x, 1] = int(new_g)
            out_img[y, x, 2] = int(new_b)
            
            # 计算量化误差
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # 扩散误差 (Floyd-Steinberg 模式，方向根据扫描方向调整)
            # 正向:      X   7          反向:  7   X
            #        3   5   1               1   5   3
            
            next_x = x + x_step
            prev_x = x - x_step
            
            # 下一个像素 (扫描方向)
            if 0 <= next_x < w:
                float_img[y, next_x, 0] += err_r * 7 / 16
                float_img[y, next_x, 1] += err_g * 7 / 16
                float_img[y, next_x, 2] += err_b * 7 / 16
                
            if y + 1 < h:
                # 下一行，扫描方向反侧
                if 0 <= prev_x < w:
                    float_img[y + 1, prev_x, 0] += err_r * 3 / 16
                    float_img[y + 1, prev_x, 1] += err_g * 3 / 16
                    float_img[y + 1, prev_x, 2] += err_b * 3 / 16
                    
                # 下一行，正下方
                float_img[y + 1, x, 0] += err_r * 5 / 16
                float_img[y + 1, x, 1] += err_g * 5 / 16
                float_img[y + 1, x, 2] += err_b * 5 / 16
                
                # 下一行，扫描方向
                if 0 <= next_x < w:
                    float_img[y + 1, next_x, 0] += err_r * 1 / 16
                    float_img[y + 1, next_x, 1] += err_g * 1 / 16
                    float_img[y + 1, next_x, 2] += err_b * 1 / 16
            
            x += x_step


class SerpentineDither(BaseDither):
    """
    蛇形 Floyd-Steinberg 误差扩散抖动 (Numba 加速版) - LAB 色彩匹配
    交替扫描方向，消除水平条纹伪影
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
        
        # 预计算 palette 的 LAB 值
        palette_lab = precompute_palette_lab(palette_float)
        
        _serpentine_fs_kernel_lab(float_img, palette_float, palette_lab, out_img)
                        
        return out_img

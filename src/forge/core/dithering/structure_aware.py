"""
结构感知抖动实现 (Numba 加速版)
使用边缘检测来自适应调整抖动强度，保留边缘清晰度
"""
import numpy as np
from numba import jit
import cv2
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
def _structure_aware_kernel(float_img, palette, out_img, edge_strength):
    """结构感知抖动核心算法 (Numba JIT 加速)"""
    h, w = float_img.shape[:2]
    
    for y in range(h):
        # 蛇形扫描
        if y % 2 == 0:
            x_range_start, x_range_end, x_step = 0, w, 1
        else:
            x_range_start, x_range_end, x_step = w - 1, -1, -1
            
        x = x_range_start
        while (x_step > 0 and x < x_range_end) or (x_step < 0 and x > x_range_end):
            old_r = float_img[y, x, 0]
            old_g = float_img[y, x, 1]
            old_b = float_img[y, x, 2]
            
            best_idx = _find_closest_color_fast(old_r, old_g, old_b, palette)
            
            new_r = palette[best_idx, 0]
            new_g = palette[best_idx, 1]
            new_b = palette[best_idx, 2]
            
            out_img[y, x, 0] = int(new_r)
            out_img[y, x, 1] = int(new_g)
            out_img[y, x, 2] = int(new_b)
            
            # 计算量化误差
            err_r = old_r - new_r
            err_g = old_g - new_g
            err_b = old_b - new_b
            
            # 根据边缘强度调整误差扩散系数
            # 边缘处减少扩散，保留清晰度
            edge_factor = 1.0 - edge_strength[y, x]
            err_r *= edge_factor
            err_g *= edge_factor
            err_b *= edge_factor
            
            # 扩散误差 (蛇形 Floyd-Steinberg)
            next_x = x + x_step
            prev_x = x - x_step
            
            if 0 <= next_x < w:
                float_img[y, next_x, 0] += err_r * 7 / 16
                float_img[y, next_x, 1] += err_g * 7 / 16
                float_img[y, next_x, 2] += err_b * 7 / 16
                
            if y + 1 < h:
                if 0 <= prev_x < w:
                    float_img[y + 1, prev_x, 0] += err_r * 3 / 16
                    float_img[y + 1, prev_x, 1] += err_g * 3 / 16
                    float_img[y + 1, prev_x, 2] += err_b * 3 / 16
                    
                float_img[y + 1, x, 0] += err_r * 5 / 16
                float_img[y + 1, x, 1] += err_g * 5 / 16
                float_img[y + 1, x, 2] += err_b * 5 / 16
                
                if 0 <= next_x < w:
                    float_img[y + 1, next_x, 0] += err_r * 1 / 16
                    float_img[y + 1, next_x, 1] += err_g * 1 / 16
                    float_img[y + 1, next_x, 2] += err_b * 1 / 16
            
            x += x_step


class StructureAwareDither(BaseDither):
    """
    结构感知抖动 (Numba 加速版)
    使用边缘检测自适应调整误差扩散，保留线条和文字清晰度
    """
    
    def __init__(self):
        super().__init__()
    
    def _compute_edge_strength(self, image: np.ndarray) -> np.ndarray:
        """计算边缘强度图 (0-1, 1=强边缘)"""
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)
            
        # Sobel 边缘检测
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 边缘幅度
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # 归一化到 0-1
        max_val = magnitude.max()
        if max_val > 0:
            magnitude = magnitude / max_val
            
        # 应用阈值使边缘更明显
        # 使用 sigmoid 风格的映射
        edge_strength = 1.0 / (1.0 + np.exp(-10 * (magnitude - 0.3)))
        
        return edge_strength.astype(np.float64)
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        h, w = image.shape[:2]
        float_img = image.astype(np.float64)
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        palette_float = palette.astype(np.float64)
        
        # 计算边缘强度
        edge_strength = self._compute_edge_strength(image)
        
        _structure_aware_kernel(float_img, palette_float, out_img, edge_strength)
                        
        return out_img

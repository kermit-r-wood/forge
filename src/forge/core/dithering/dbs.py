"""
Direct Binary Search (DBS) 抖动实现 (Numba 部分加速)
迭代优化算法，质量最高但速度较慢
用于 "超高画质" 模式
"""
import numpy as np
from numba import jit, prange
import cv2
from .base import BaseDither


@jit(nopython=True, cache=True)
def _find_closest_color_idx(pixel_r, pixel_g, pixel_b, palette):
    """快速查找最近颜色索引 (Numba JIT)"""
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
def _compute_local_error(out_img, target_img, filter_kernel, y, x, h, w):
    """计算局部加权误差 (使用 HVS 滤波器)"""
    kh, kw = filter_kernel.shape
    half_kh = kh // 2
    half_kw = kw // 2
    
    error = 0.0
    
    for ky in range(-half_kh, half_kh + 1):
        for kx in range(-half_kw, half_kw + 1):
            ny = y + ky
            nx = x + kx
            
            if 0 <= ny < h and 0 <= nx < w:
                weight = filter_kernel[ky + half_kh, kx + half_kw]
                
                for c in range(3):
                    diff = float(out_img[ny, nx, c]) - float(target_img[ny, nx, c])
                    error += weight * diff * diff
                    
    return error


@jit(nopython=True, parallel=True, cache=True)
def _initialize_output(target_img, palette, out_img, indices):
    """初始化输出图像 (并行)"""
    h, w = target_img.shape[:2]
    
    for y in prange(h):
        for x in range(w):
            r = target_img[y, x, 0]
            g = target_img[y, x, 1]
            b = target_img[y, x, 2]
            
            idx = _find_closest_color_idx(r, g, b, palette)
            indices[y, x] = idx
            out_img[y, x, 0] = palette[idx, 0]
            out_img[y, x, 1] = palette[idx, 1]
            out_img[y, x, 2] = palette[idx, 2]


@jit(nopython=True, cache=True)
def _try_swap(out_img, target_img, indices, palette, filter_kernel, y, x, h, w, n_colors):
    """尝试交换当前像素的颜色，如果能降低误差则接受"""
    current_idx = indices[y, x]
    current_error = _compute_local_error(out_img, target_img, filter_kernel, y, x, h, w)
    
    best_idx = current_idx
    best_error = current_error
    
    # 尝试所有其他颜色
    for new_idx in range(n_colors):
        if new_idx == current_idx:
            continue
            
        # 临时替换
        old_r = out_img[y, x, 0]
        old_g = out_img[y, x, 1]
        old_b = out_img[y, x, 2]
        
        out_img[y, x, 0] = palette[new_idx, 0]
        out_img[y, x, 1] = palette[new_idx, 1]
        out_img[y, x, 2] = palette[new_idx, 2]
        
        new_error = _compute_local_error(out_img, target_img, filter_kernel, y, x, h, w)
        
        if new_error < best_error:
            best_error = new_error
            best_idx = new_idx
            
        # 恢复
        out_img[y, x, 0] = old_r
        out_img[y, x, 1] = old_g
        out_img[y, x, 2] = old_b
    
    # 如果找到更好的，应用它
    if best_idx != current_idx:
        indices[y, x] = best_idx
        out_img[y, x, 0] = palette[best_idx, 0]
        out_img[y, x, 1] = palette[best_idx, 1]
        out_img[y, x, 2] = palette[best_idx, 2]
        return True
        
    return False


@jit(nopython=True, cache=True)
def _dbs_iteration(out_img, target_img, indices, palette, filter_kernel, h, w, n_colors):
    """执行一轮 DBS 迭代"""
    changes = 0
    
    for y in range(h):
        for x in range(w):
            if _try_swap(out_img, target_img, indices, palette, filter_kernel, y, x, h, w, n_colors):
                changes += 1
                
    return changes


class DBSDither(BaseDither):
    """
    Direct Binary Search (DBS) 抖动
    迭代优化算法，提供最高质量但速度较慢
    适用于追求极致画质的场景
    """
    
    def __init__(self, max_iterations: int = 5):
        super().__init__()
        self.max_iterations = max_iterations
        self._filter_kernel = None
        
    def _create_hvs_filter(self, size: int = 5) -> np.ndarray:
        """创建人眼视觉系统 (HVS) 加权滤波器"""
        # 简化的高斯形式 HVS 滤波器
        sigma = size / 3.0
        kernel = np.zeros((size, size), dtype=np.float64)
        center = size // 2
        
        for y in range(size):
            for x in range(size):
                dist_sq = (y - center)**2 + (x - center)**2
                kernel[y, x] = np.exp(-dist_sq / (2 * sigma**2))
                
        # 归一化
        kernel /= kernel.sum()
        return kernel
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        h, w = image.shape[:2]
        target_img = image.astype(np.float64)
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        indices = np.zeros((h, w), dtype=np.int32)
        palette_uint8 = palette.astype(np.uint8)
        n_colors = len(palette)
        
        # 创建 HVS 滤波器
        if self._filter_kernel is None:
            self._filter_kernel = self._create_hvs_filter(5)
        filter_kernel = self._filter_kernel
        
        # 初始化 (最近邻)
        _initialize_output(target_img, palette_uint8, out_img, indices)
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            changes = _dbs_iteration(
                out_img, target_img, indices, palette_uint8, 
                filter_kernel, h, w, n_colors
            )
            
            # 如果没有改进，提前终止
            if changes == 0:
                break
                        
        return out_img

"""
Riemersma 抖动实现 (基于 Hilbert 曲线) - LAB 色彩空间
沿空间填充曲线进行误差扩散，消除方向性伪影
"""
import numpy as np
from numba import jit
from .base import BaseDither, _find_closest_color_lab, precompute_palette_lab


@jit(nopython=True, cache=False)
def _hilbert_d2xy(n, d):
    """将 Hilbert 曲线上的位置 d 转换为 (x, y) 坐标"""
    x = 0
    y = 0
    s = 1
    while s < n:
        rx = 1 & (d // 2)
        ry = 1 & (d ^ rx)
        
        # 旋转
        if ry == 0:
            if rx == 1:
                x = s - 1 - x
                y = s - 1 - y
            x, y = y, x
            
        x += s * rx
        y += s * ry
        d //= 4
        s *= 2
    return x, y


@jit(nopython=True, cache=False)
def _generate_hilbert_path(size):
    """生成 Hilbert 曲线路径"""
    n = size * size
    path = np.zeros((n, 2), dtype=np.int32)
    for i in range(n):
        x, y = _hilbert_d2xy(size, i)
        path[i, 0] = x
        path[i, 1] = y
    return path


@jit(nopython=True, cache=False)
def _riemersma_kernel_lab(float_img, palette_rgb, palette_lab, out_img, path, weights):
    """Riemersma 抖动核心算法 - 使用 LAB 色彩匹配"""
    h, w = float_img.shape[:2]
    n_path = len(path)
    queue_size = len(weights)
    
    # 误差队列 (用于沿曲线传播误差)
    err_queue_r = np.zeros(queue_size, dtype=np.float64)
    err_queue_g = np.zeros(queue_size, dtype=np.float64)
    err_queue_b = np.zeros(queue_size, dtype=np.float64)
    
    for i in range(n_path):
        x = path[i, 0]
        y = path[i, 1]
        
        if x >= w or y >= h:
            continue
            
        # 应用累积误差
        total_weight = 0.0
        added_r = 0.0
        added_g = 0.0
        added_b = 0.0
        
        for j in range(queue_size):
            added_r += err_queue_r[j] * weights[j]
            added_g += err_queue_g[j] * weights[j]
            added_b += err_queue_b[j] * weights[j]
            total_weight += weights[j]
            
        if total_weight > 0:
            added_r /= total_weight
            added_g /= total_weight
            added_b /= total_weight
        
        old_r = float_img[y, x, 0] + added_r
        old_g = float_img[y, x, 1] + added_g
        old_b = float_img[y, x, 2] + added_b
        
        best_idx = _find_closest_color_lab(old_r, old_g, old_b, palette_lab)
        
        new_r = palette_rgb[best_idx, 0]
        new_g = palette_rgb[best_idx, 1]
        new_b = palette_rgb[best_idx, 2]
        
        out_img[y, x, 0] = int(max(0, min(255, new_r)))
        out_img[y, x, 1] = int(max(0, min(255, new_g)))
        out_img[y, x, 2] = int(max(0, min(255, new_b)))
        
        # 计算误差并推入队列
        err_r = old_r - new_r
        err_g = old_g - new_g
        err_b = old_b - new_b
        
        # 移动队列
        for j in range(queue_size - 1, 0, -1):
            err_queue_r[j] = err_queue_r[j - 1]
            err_queue_g[j] = err_queue_g[j - 1]
            err_queue_b[j] = err_queue_b[j - 1]
            
        err_queue_r[0] = err_r
        err_queue_g[0] = err_g
        err_queue_b[0] = err_b


class RiemersmaDither(BaseDither):
    """
    Riemersma 抖动 (基于 Hilbert 曲线) - LAB 色彩匹配
    沿空间填充曲线进行误差扩散，无方向性伪影
    """
    
    def __init__(self):
        super().__init__()
        self._path_cache = {}
        
    def _get_hilbert_size(self, w, h):
        """获取适合的 Hilbert 曲线大小 (必须是 2 的幂)"""
        max_dim = max(w, h)
        size = 1
        while size < max_dim:
            size *= 2
        return size
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        h, w = image.shape[:2]
        float_img = image.astype(np.float64)
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        palette_float = palette.astype(np.float64)
        
        # 预计算 palette 的 LAB 值
        palette_lab = precompute_palette_lab(palette_float)
        
        # 获取或生成 Hilbert 路径
        hilbert_size = self._get_hilbert_size(w, h)
        cache_key = hilbert_size
        
        if cache_key not in self._path_cache:
            self._path_cache[cache_key] = _generate_hilbert_path(hilbert_size)
            
        path = self._path_cache[cache_key]
        
        # 权重衰减 (指数衰减)
        queue_size = 16
        weights = np.array([2.0 ** (-(i / 4.0)) for i in range(queue_size)], dtype=np.float64)
        
        _riemersma_kernel_lab(float_img, palette_float, palette_lab, out_img, path, weights)
                        
        return out_img

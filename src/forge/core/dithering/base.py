"""
抖动算法基类
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2
from numba import jit


# ============================================================
# Shared LAB Color Matching Functions (Numba JIT compatible)
# ============================================================

@jit(nopython=True, cache=True)
def _rgb_to_lab_fast(r, g, b):
    """
    Fast RGB to LAB conversion (Numba JIT compatible)
    Uses simplified sRGB linearization for speed
    """
    # Normalize to 0-1
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    # Simplified sRGB linearization
    if r <= 0.04045:
        r = r / 12.92
    else:
        r = ((r + 0.055) / 1.055) ** 2.4
    
    if g <= 0.04045:
        g = g / 12.92
    else:
        g = ((g + 0.055) / 1.055) ** 2.4
    
    if b <= 0.04045:
        b = b / 12.92
    else:
        b = ((b + 0.055) / 1.055) ** 2.4
    
    # RGB to XYZ (D65 illuminant)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    # Normalize by D65 white point
    x = x / 0.95047
    y = y / 1.0
    z = z / 1.08883
    
    # f(t) function for LAB
    epsilon = 0.008856
    kappa = 903.3
    
    if x > epsilon:
        fx = x ** (1.0 / 3.0)
    else:
        fx = (kappa * x + 16.0) / 116.0
    
    if y > epsilon:
        fy = y ** (1.0 / 3.0)
    else:
        fy = (kappa * y + 16.0) / 116.0
    
    if z > epsilon:
        fz = z ** (1.0 / 3.0)
    else:
        fz = (kappa * z + 16.0) / 116.0
    
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b_val = 200.0 * (fy - fz)
    
    return L, a, b_val


@jit(nopython=True, cache=True)
def _find_closest_color_lab(pixel_r, pixel_g, pixel_b, palette_lab):
    """
    Find closest palette color using LAB Euclidean distance (CIE76)
    
    Args:
        pixel_r, pixel_g, pixel_b: RGB values (0-255 range, can be float)
        palette_lab: Pre-computed LAB values of palette (N, 3)
    
    Returns:
        Index of closest color in palette
    """
    # Clip inputs to 0-255 range to avoid LAB conversion issues with out-of-bounds values
    # originating from error diffusion mechanisms
    r = min(255.0, max(0.0, float(pixel_r)))
    g = min(255.0, max(0.0, float(pixel_g)))
    b = min(255.0, max(0.0, float(pixel_b)))
    
    # Convert pixel to LAB
    pL, pa, pb = _rgb_to_lab_fast(r, g, b)
    
    best_dist = 1e10
    best_idx = 0
    
    for i in range(len(palette_lab)):
        dL = pL - palette_lab[i, 0]
        da = pa - palette_lab[i, 1]
        db = pb - palette_lab[i, 2]
        dist = dL * dL + da * da + db * db
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    
    return best_idx


@jit(nopython=True, cache=True)
def precompute_palette_lab(palette_rgb):
    """
    Pre-compute LAB values for entire palette
    
    Args:
        palette_rgb: RGB palette (N, 3) uint8 or float64
    
    Returns:
        LAB palette (N, 3) float64
    """
    n = len(palette_rgb)
    palette_lab = np.zeros((n, 3), dtype=np.float64)
    
    for i in range(n):
        L, a, b = _rgb_to_lab_fast(
            palette_rgb[i, 0], 
            palette_rgb[i, 1], 
            palette_rgb[i, 2]
        )
        palette_lab[i, 0] = L
        palette_lab[i, 1] = a
        palette_lab[i, 2] = b
    
    return palette_lab

class BaseDither(ABC):
    """抖动算法基类"""
    
    def __init__(self, distance_metric='cie76'):
        self._palette_lab = None  # 缓存 LAB 调色板
        self._palette_rgb = None
        self._kdtree = None
        self.distance_metric = distance_metric
    
    @abstractmethod
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """
        应用抖动算法
        :param image: 输入图像 (H, W, 3) float/uint8
        :param palette: 调色板 (N, 3) uint8
        :return: 抖动后的索引图或RGB图
        """
        pass
    
    def _ensure_lab_palette(self, palette: np.ndarray):
        """确保 LAB 调色板已计算"""
        if self._palette_rgb is None or not np.array_equal(self._palette_rgb, palette):
            self._palette_rgb = palette.copy()
            # 转换 palette 到 LAB
            palette_reshaped = palette.reshape(1, -1, 3).astype(np.float32) / 255.0
            self._palette_lab = cv2.cvtColor(palette_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            
            # 构建 KDTree 用于快速粗略查找
            from scipy.spatial import KDTree
            self._kdtree = KDTree(self._palette_lab)
        
    def find_closest_color(self, pixel: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """使用选定的距离算法找到最接近的颜色"""
        self._ensure_lab_palette(palette)
        
        # 将单个像素转换为 LAB
        pixel_rgb = (np.clip(pixel, 0, 255).astype(np.float32) / 255.0).reshape(1, 1, 3)
        pixel_lab = cv2.cvtColor(pixel_rgb, cv2.COLOR_RGB2LAB).reshape(3).astype(np.float32)
        
        # 如果是 CIE76 (Euclidean)，直接使用 KDTree
        if self.distance_metric == 'cie76':
            _, idx = self._kdtree.query(pixel_lab)
            return palette[idx]
            
        # 对于复杂算法 (CIEDE2000, CIE94)，使用混合策略
        # 1. 使用 CIE76 找到最近的 N 个邻居
        k = 20 # 候选数量
        _, candidates_idx = self._kdtree.query(pixel_lab, k=min(k, len(palette)))
        
        if isinstance(candidates_idx, int):
             candidates_idx = [candidates_idx]
             
        candidates_lab = self._palette_lab[candidates_idx]
        
        # 2. 计算精确距离
        # 2. 计算精确距离
        from ..color_distance import ciede2000_distance
        
        if self.distance_metric == 'ciede2000':
            dists = ciede2000_distance(pixel_lab, candidates_lab)
        else:
             # Fallback to Euclidean
             diff = candidates_lab - pixel_lab
             dists = np.sum(diff**2, axis=1)
             
        best_idx_in_candidates = np.argmin(dists)
        best_global_idx = candidates_idx[best_idx_in_candidates]
        
        return palette[best_global_idx]

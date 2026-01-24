"""
抖动算法基类
"""
from abc import ABC, abstractmethod
import numpy as np
import cv2

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
            palette_reshaped = palette.reshape(1, -1, 3).astype(np.uint8)
            self._palette_lab = cv2.cvtColor(palette_reshaped, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            
            # 构建 KDTree 用于快速粗略查找
            from scipy.spatial import KDTree
            self._kdtree = KDTree(self._palette_lab)
        
    def find_closest_color(self, pixel: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """使用选定的距离算法找到最接近的颜色"""
        self._ensure_lab_palette(palette)
        
        # 将单个像素转换为 LAB
        pixel_rgb = np.clip(pixel, 0, 255).astype(np.uint8).reshape(1, 1, 3)
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

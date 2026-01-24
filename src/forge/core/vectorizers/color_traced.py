"""
Color-Traced 矢量化实现
基于颜色分层 + 轮廓提取 + 形态学处理
"""
import numpy as np
import cv2
from .base import BaseVectorizer


class ColorTracedVectorizer(BaseVectorizer):
    """
    Color-Traced 矢量化
    适用于动漫、插画、Logo 等扁平风格图像
    """
    
    def __init__(self, n_colors: int = 16):
        """
        Args:
            n_colors: K-Means 量化颜色数量
        """
        self.n_colors = n_colors
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行 Color-Traced 矢量化"""
        h, w = image.shape[:2]
        
        # 1. K-Means 量化减色
        quantized = self._quantize_image(image, self.n_colors)
        
        # 2. 映射到目标调色板
        indices = self._map_to_palette(quantized, palette)
        
        # 3. 分层绘制 (大区域先绘制，小区域后绘制)
        result = self._draw_layers(indices, palette, h, w)
        
        return result, indices
    
    def _quantize_image(self, image: np.ndarray, n_colors: int) -> np.ndarray:
        """使用 K-Means 量化图像颜色"""
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(h, w, 3)
        return quantized
    
    def _map_to_palette(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """将图像颜色映射到调色板，返回索引图"""
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3).astype(np.float32)
        palette_f = palette.astype(np.float32)
        
        # 分批处理避免内存溢出
        batch_size = 50000
        indices = np.zeros(len(pixels), dtype=np.int32)
        
        for i in range(0, len(pixels), batch_size):
            batch = pixels[i:i+batch_size]
            diff = batch[:, np.newaxis, :] - palette_f[np.newaxis, :, :]
            dist = np.sum(diff ** 2, axis=2)
            indices[i:i+batch_size] = np.argmin(dist, axis=1)
        
        return indices.reshape(h, w)
    
    def _draw_layers(self, indices: np.ndarray, palette: np.ndarray, 
                     h: int, w: int) -> np.ndarray:
        """分层绘制：按区域面积从大到小"""
        result = np.zeros((h, w, 3), dtype=np.uint8)
        unique_indices = np.unique(indices)
        
        # 收集区域信息
        regions = []
        for idx in unique_indices:
            mask = (indices == idx).astype(np.uint8) * 255
            area = np.sum(mask > 0)
            regions.append((int(idx), mask, area))
        
        # 按面积从大到小排序
        regions.sort(key=lambda x: x[2], reverse=True)
        
        # 绘制每个区域
        for palette_idx, mask, _ in regions:
            # 形态学处理去噪
            kernel = np.ones((2, 2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 找轮廓
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # 用调色板颜色填充
            color = tuple(int(c) for c in palette[palette_idx])
            cv2.drawContours(result, contours, -1, color, -1)
        
        return result

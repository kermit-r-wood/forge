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
        indices_raw = self._map_to_palette(quantized, palette)
        
        # 3. 分层绘制 (大区域先绘制，小区域后绘制)
        # Returns: (smoothed_rgb, smoothed_indices)
        result, indices_smooth = self._draw_layers(indices_raw, palette, h, w)
        
        return result, indices_smooth
    
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
                     h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Post-process indices to remove thin lines ("cantilevers") and small floating islands.
        Optimized for 3D printing to prevent printer head clogging from small parts.
        """
        # Parameters - increased for 3D printing optimization
        MIN_AREA = 100     # Filter out regions smaller than this (area size) - increased from 25
        KERNEL_SIZE = 3    # Kernel for morphological opening (removes thin lines < 3px)
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)  # Larger kernel for closing
        
        # 1. Determine Background Index (Most frequent color)
        # This will be used to fill voids created by filtering
        counts = np.bincount(indices.flatten())
        bg_index = np.argmax(counts)
        
        # Start with a copy of raw indices
        result_indices = indices.copy().astype(np.int32)
        unique_indices = np.unique(indices)
        
        # 2. Filter Process for each color
        for idx in unique_indices:
            if idx == bg_index:
                continue
                
            # Create binary mask for current color
            mask = (indices == idx).astype(np.uint8) * 255
            
            # A. Morphological Opening
            # Removes thin connections and small noise
            mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # B. Morphological Closing
            # Fills small holes and connects nearby regions to reduce fragmentation
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
            
            # C. Erosion to remove thin protrusions (followed by dilation to restore size)
            # This removes thin "fingers" that can cause printer head issues
            mask_eroded = cv2.erode(mask_closed, kernel, iterations=1)
            mask_cleaned = cv2.dilate(mask_eroded, kernel, iterations=1)
            
            # Identify pixels removed by filtering and set them to background
            pixels_removed = (mask > 0) & (mask_cleaned == 0)
            result_indices[pixels_removed] = bg_index
            
            # D. Connected Component Analysis
            # Filter out individual disconnected islands that are too small
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
            
            # Iterate through components (label 0 is background of the mask, ignore it)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < MIN_AREA:
                    # Filter out this small component
                    component_mask = (labels == i)
                    result_indices[component_mask] = bg_index
        
        # 3. Generate RGB Result
        # Direct mapping ensures the preview exactly matches the indices
        result_rgb = palette[result_indices].astype(np.uint8)
        
        # Optional: Add anti-aliasing just for the preview (result_rgb)?
        # For now, let's keep it pixel-perfect to avoid misleading the user 
        # about what will actually be printed.
        
        return result_rgb, result_indices

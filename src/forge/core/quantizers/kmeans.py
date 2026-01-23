"""
K-Means 量化器
"""
import numpy as np
from sklearn.cluster import KMeans
from .base import BaseQuantizer

class KMeansQuantizer(BaseQuantizer):
    """K-Means 聚类量化器"""
    
    def quantize(self, image: np.ndarray, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
        if image is None:
            return None, None
            
        h, w, c = image.shape
        pixels = image.reshape(-1, c)
        
        # 使用 K-Means 聚类
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        
        # 重建图像
        quantized_pixels = palette[labels]
        quantized_image = quantized_pixels.reshape(h, w, c).astype(np.uint8)
        
        return quantized_image, palette

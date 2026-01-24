"""
Vectorizer 基类
"""
import numpy as np
from abc import ABC, abstractmethod


class BaseVectorizer(ABC):
    """矢量化处理器基类"""
    
    @abstractmethod
    def apply(self, image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        执行矢量化处理
        
        Args:
            image: 输入图像 (H, W, 3) RGB
            palette: 目标调色板 (N, 3) RGB
            
        Returns:
            tuple: (processed_image, indices)
                - processed_image: 处理后的 RGB 图像 (H, W, 3)
                - indices: 调色板索引图 (H, W)
        """
        pass

"""
量化器基类
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseQuantizer(ABC):
    """色彩量化器基类"""
    
    @abstractmethod
    def quantize(self, image: np.ndarray, n_colors: int) -> tuple[np.ndarray, np.ndarray]:
        """
        量化图像
        :param image: 输入图像 (H, W, 3) uint8
        :param n_colors: 目标颜色数量
        :return: (量化后的图像, 调色板)
        """
        pass

"""
双边滤波器实现
"""
import cv2
import numpy as np
from .base import BaseFilter

class BilateralFilter(BaseFilter):
    """
    双边滤波器
    同时考虑空间邻近度和像素值相似度，在去噪的同时保留边缘
    """
    
    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        """
        :param d: 滤波过程中使用的每个像素邻域的直径
        :param sigma_color: 颜色空间的标准差
        :param sigma_space: 坐标空间的标准差
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        # 确保是 BGR 格式 (OpenCV 默认) 或 RGB
        # cv2.bilateralFilter 支持 8位或浮点型图像
        return cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)

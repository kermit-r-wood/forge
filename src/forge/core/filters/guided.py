"""
引导滤波器实现
Reference: He, K., Sun, J., & Tang, X. (2010). Guided image filtering.
"""
import cv2
import numpy as np
from .base import BaseFilter

class GuidedFilter(BaseFilter):
    """
    引导滤波器
    """
    
    def __init__(self, radius: int = 4, eps: float = 0.01):
        """
        :param radius: 滤波半径
        :param eps: 正则化参数 epsilon
        """
        self.radius = radius
        self.eps = eps
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        # 归一化到 [0, 1]
        img_float = image.astype(np.float32) / 255.0
        
        # 引导滤波需要引导图，这里使用图像自身作为引导图
        result = self._guided_filter(img_float, img_float, self.radius, self.eps)
        
        # 还原到 [0, 255]
        return np.clip(result * 255, 0, 255).astype(np.uint8)

    def _guided_filter(self, I, p, r, eps):
        """
        I: 引导图 (float32)
        p: 输入图 (float32)
        r: 半径
        eps: 正则化参数
        """
        # 使用 OpenCV 的 boxFilter 加速计算
        mean_I = cv2.boxFilter(I, cv2.CV_32F, (r, r))
        mean_p = cv2.boxFilter(p, cv2.CV_32F, (r, r))
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

        q = mean_a * I + mean_b
        return q

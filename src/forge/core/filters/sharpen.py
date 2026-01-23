
import cv2
import numpy as np
from .base import BaseFilter

class SharpenFilter(BaseFilter):
    """
    USM 锐化滤镜
    提升图像细节，对抗量化造成的模糊
    """
    def __init__(self, amount=1.5, radius=1.0, threshold=0):
        self.amount = amount
        self.radius = radius
        self.threshold = threshold
        
    def apply(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return None
            
        # 转换到 float 做计算
        img_float = image.astype(np.float32)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(img_float, (0, 0), self.radius)
        
        # 锐化: Original + (Original - Blurred) * Amount
        sharpened = img_float + (img_float - blurred) * self.amount
        
        # Clip back to 0-255
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened

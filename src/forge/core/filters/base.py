"""
抽象滤波器基类
"""
from abc import ABC, abstractmethod
import numpy as np

class BaseFilter(ABC):
    """滤波器基类"""
    
    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """应用滤波器"""
        pass

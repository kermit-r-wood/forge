"""
Blue Noise Dithering 实现
使用 Void-and-Cluster 生成的阈值矩阵实现高质量抖动
比 Floyd-Steinberg 产生更自然均匀的噪点分布
"""
import numpy as np
from numba import jit, prange
from .base import BaseDither
import cv2

# 预计算的 64x64 Blue Noise 阈值矩阵 (Void-and-Cluster 算法生成)
# 值范围 0-4095，将在运行时归一化
_BLUE_NOISE_64 = None

def _generate_blue_noise_matrix(size: int = 64) -> np.ndarray:
    """
    生成 Blue Noise 阈值矩阵 (简化的 Void-and-Cluster 算法)
    实际应用中可以使用预计算的高质量矩阵
    """
    np.random.seed(42)  # 确保可重复
    
    # 使用简化方法：从随机噪声开始，通过高斯模糊迭代优化
    matrix = np.random.rand(size, size).astype(np.float32)
    
    # 迭代优化使分布更均匀
    for _ in range(50):
        # 找到最密集的点（周围值最高）和最稀疏的点（周围值最低）
        blurred = cv2.GaussianBlur(matrix, (0, 0), 1.5)
        
        # 找到最大和最小位置
        max_pos = np.unravel_index(np.argmax(blurred), blurred.shape)
        min_pos = np.unravel_index(np.argmin(blurred), blurred.shape)
        
        # 交换这两个点的值
        matrix[max_pos], matrix[min_pos] = matrix[min_pos], matrix[max_pos]
    
    # 归一化到 0-1
    matrix = (matrix - matrix.min()) / (matrix.max() - matrix.min())
    return matrix

def _get_blue_noise_matrix() -> np.ndarray:
    """获取或生成 Blue Noise 矩阵"""
    global _BLUE_NOISE_64
    if _BLUE_NOISE_64 is None:
        _BLUE_NOISE_64 = _generate_blue_noise_matrix(64)
    return _BLUE_NOISE_64


@jit(nopython=True, parallel=True, cache=True)
def _blue_noise_dither_kernel(
    image: np.ndarray,
    palette_lab: np.ndarray, 
    palette_rgb: np.ndarray,
    threshold_matrix: np.ndarray,
    out_img: np.ndarray
):
    """
    Blue Noise 抖动核心算法 (Numba JIT 加速)
    使用阈值矩阵决定是否选择较亮或较暗的调色板颜色
    """
    h, w, _ = image.shape
    th, tw = threshold_matrix.shape
    n_colors = len(palette_lab)
    
    for y in prange(h):
        for x in range(w):
            pixel = image[y, x]
            
            # 转换到 LAB (简化：使用 RGB 近似)
            # 完整实现应使用真正的 LAB 转换
            r, g, b = pixel[0], pixel[1], pixel[2]
            
            # 找到两个最近的调色板颜色
            best_dist = 1e10
            second_dist = 1e10
            best_idx = 0
            second_idx = 0
            
            for i in range(n_colors):
                pr, pg, pb = palette_rgb[i, 0], palette_rgb[i, 1], palette_rgb[i, 2]
                dist = (r - pr)**2 + (g - pg)**2 + (b - pb)**2
                
                if dist < best_dist:
                    second_dist = best_dist
                    second_idx = best_idx
                    best_dist = dist
                    best_idx = i
                elif dist < second_dist:
                    second_dist = dist
                    second_idx = i
            
            # 使用 Blue Noise 阈值决定选择哪个颜色
            threshold = threshold_matrix[y % th, x % tw]
            
            # 计算两个颜色之间的权重
            if best_dist + second_dist > 0:
                weight = best_dist / (best_dist + second_dist)
            else:
                weight = 0.0
            
            # 如果阈值大于权重，选择次优颜色（产生抖动效果）
            if threshold > weight:
                chosen_idx = best_idx
            else:
                chosen_idx = second_idx
            
            out_img[y, x, 0] = palette_rgb[chosen_idx, 0]
            out_img[y, x, 1] = palette_rgb[chosen_idx, 1]
            out_img[y, x, 2] = palette_rgb[chosen_idx, 2]


class BlueNoiseDither(BaseDither):
    """
    Blue Noise 抖动算法
    产生自然均匀的噪点分布，视觉效果优于 Floyd-Steinberg
    """
    
    def __init__(self):
        super().__init__()
        self._threshold_matrix = None
    
    def _ensure_threshold_matrix(self):
        if self._threshold_matrix is None:
            self._threshold_matrix = _get_blue_noise_matrix()
    
    def apply(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        
        self._ensure_threshold_matrix()
        self._ensure_lab_palette(palette)
        
        h, w = image.shape[:2]
        out_img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 准备数据
        img_float = image.astype(np.float32)
        palette_float = palette.astype(np.float32)
        
        # 调用 Numba 加速的核心算法
        _blue_noise_dither_kernel(
            img_float,
            self._palette_lab,
            palette_float,
            self._threshold_matrix,
            out_img
        )
        
        return out_img

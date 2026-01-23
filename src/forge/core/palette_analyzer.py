"""
调色板覆盖度分析器
分析源图像颜色分布，评估调色板对其的覆盖程度
"""
import numpy as np
import cv2
from scipy.spatial import KDTree


class PaletteAnalyzer:
    """
    分析图像颜色与调色板的匹配程度
    生成覆盖度热力图，显示哪些区域的颜色无法被调色板准确表达
    """
    
    def __init__(self, palette: np.ndarray):
        """
        Args:
            palette: 调色板颜色数组 (N, 3) RGB
        """
        self.palette = palette
        
        # 转换到 LAB 空间
        palette_rgb = palette.reshape(1, -1, 3).astype(np.uint8)
        self.palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        
        # 构建 KDTree
        self.tree = KDTree(self.palette_lab)
    
    def analyze(self, image: np.ndarray) -> dict:
        """
        分析图像的调色板覆盖度
        
        Args:
            image: RGB 图像 (H, W, 3)
            
        Returns:
            dict: 包含以下键:
                - distance_map: 每个像素到最近调色板颜色的距离 (H, W)
                - heatmap: 可视化热力图 (H, W, 3) RGB，红色表示覆盖差
                - stats: 统计信息字典
        """
        # 转换到 LAB
        image_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        h, w = image.shape[:2]
        
        # 计算每个像素到最近调色板颜色的距离
        flat_lab = image_lab.reshape(-1, 3)
        distances, indices = self.tree.query(flat_lab)
        
        distance_map = distances.reshape(h, w)
        
        # 生成热力图
        heatmap = self._create_heatmap(distance_map)
        
        # 统计信息
        stats = {
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'mean_distance': float(np.mean(distances)),
            'median_distance': float(np.median(distances)),
            'std_distance': float(np.std(distances)),
            'poor_coverage_percent': float(np.mean(distances > 20) * 100),  # 距离 > 20 视为覆盖差
            'good_coverage_percent': float(np.mean(distances < 10) * 100),  # 距离 < 10 视为覆盖好
        }
        
        return {
            'distance_map': distance_map,
            'heatmap': heatmap,
            'stats': stats
        }
    
    def _create_heatmap(self, distance_map: np.ndarray) -> np.ndarray:
        """
        将距离图转换为可视化热力图
        绿色 = 好 (距离小)
        黄色 = 中等
        红色 = 差 (距离大)
        """
        # 归一化到 0-1 (假设最大可接受距离为 50)
        normalized = np.clip(distance_map / 50.0, 0, 1)
        
        # 创建 HSV 图像 (Hue: 120=绿 -> 60=黄 -> 0=红)
        h, w = distance_map.shape
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Hue: 从 120 (绿) 到 0 (红) 线性插值
        hsv[:, :, 0] = ((1 - normalized) * 60).astype(np.uint8)  # 0=红, 60=绿
        hsv[:, :, 1] = 200  # 饱和度
        hsv[:, :, 2] = 255  # 亮度
        
        # 转换到 RGB
        heatmap = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        return heatmap
    
    def get_problem_colors(self, image: np.ndarray, threshold: float = 25, top_n: int = 10) -> list:
        """
        获取调色板覆盖最差的颜色
        
        Args:
            image: RGB 图像
            threshold: 距离阈值，超过此值视为"问题颜色"
            top_n: 返回前 N 个最差的颜色
            
        Returns:
            list of dict: 每个 dict 包含 'color' (RGB), 'distance', 'count'
        """
        image_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        flat_lab = image_lab.reshape(-1, 3)
        flat_rgb = image.reshape(-1, 3)
        
        distances, _ = self.tree.query(flat_lab)
        
        # 找到问题像素
        problem_mask = distances > threshold
        problem_rgb = flat_rgb[problem_mask]
        problem_distances = distances[problem_mask]
        
        if len(problem_rgb) == 0:
            return []
        
        # 简单聚类问题颜色 (按 RGB 分箱)
        # 量化到 8 级
        quantized = (problem_rgb // 32).astype(np.int32)
        unique_colors, inverse, counts = np.unique(
            quantized, axis=0, return_inverse=True, return_counts=True
        )
        
        # 计算每个聚类的平均距离
        cluster_distances = np.zeros(len(unique_colors))
        for i in range(len(unique_colors)):
            cluster_distances[i] = np.mean(problem_distances[inverse == i])
        
        # 按距离排序
        sorted_idx = np.argsort(-cluster_distances)[:top_n]
        
        results = []
        for idx in sorted_idx:
            # 还原代表颜色 (聚类中心 * 32 + 16)
            representative_color = unique_colors[idx] * 32 + 16
            results.append({
                'color': representative_color.tolist(),
                'distance': float(cluster_distances[idx]),
                'count': int(counts[idx])
            })
        
        return results

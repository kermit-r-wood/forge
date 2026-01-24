"""
Analyzer 图像分析核心
"""
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

from .filters.bilateral import BilateralFilter
from .filters.guided import GuidedFilter
from .filters.sharpen import SharpenFilter
from .quantizers.kmeans import KMeansQuantizer
from .quantizers.median_cut import MedianCutQuantizer
from .quantizers.octree import OctreeQuantizer
from .dithering.floyd_steinberg import FloydSteinbergDither
from .dithering.atkinson import AtkinsonDither
from .dithering.sierra import SierraDither
from .dithering.blue_noise import BlueNoiseDither
from .dithering.ordered import OrderedDither
from .dithering.serpentine import SerpentineDither
from .dithering.riemersma import RiemersmaDither
from .dithering.structure_aware import StructureAwareDither
from .dithering.dbs import DBSDither
from .color_model import ColorModel
from .vectorizers.color_traced import ColorTracedVectorizer
from .vectorizers.vtracer_wrapper import VTracerVectorizer

class Analyzer:
    """图像分析控制器"""
    
    def __init__(self):
        self.image = None       # 原始图像 (RGB)
        self.processed = None   # 处理后图像 (RGB)
        self.indices = None     # 索引图 (H, W) -> Palette Index
        self.palette = None     # 当前使用的 Palette (from ColorModel)
        self.combinations = None # Palette 对应的层组合
        
        # 算法实例
        self.filters = {
            0: BilateralFilter(),
            1: GuidedFilter(),
            2: None, # 无
            3: SharpenFilter()
        }
        self.quantizers = {
            0: KMeansQuantizer(),
            1: MedianCutQuantizer(),
            2: OctreeQuantizer(),
            3: None # 无 (直接使用原图颜色)
        }
        self.dithers = {
            0: FloydSteinbergDither(),
            1: AtkinsonDither(),
            2: SierraDither(),
            3: None, # 无
            4: BlueNoiseDither(),
            5: OrderedDither(),
            6: SerpentineDither(),       # 蛇形 FS
            7: RiemersmaDither(),        # Hilbert 曲线
            8: StructureAwareDither(),   # 结构感知
            9: DBSDither()               # DBS 极致画质
        }
        # 矢量化处理器
        self.vectorizers = {
            0: None,                      # 不使用矢量化
            1: ColorTracedVectorizer(),   # Color-Traced (推荐)
            2: VTracerVectorizer()        # VTracer (可选)
        }
        
    def _update_dither_settings(self, metric: str):
        """更新抖动算法设置"""
        for dither in self.dithers.values():
            if dither:
                dither.distance_metric = metric
    
    def load_image(self, path: str | Path, size: int = 100):
        """加载并调整图像大小"""
        # 读取图像
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("无法读取图像")
        
        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小 (保持比例，最长边 = size 像素? 或者 size mm * resolution?)
        # 这里 size 参数假设是 mm? 不，analyzer 的 size 通常指像素尺寸
        # 如果 size 是 3MF 输出尺寸(mm)，我们需要决定像素密度(resolution)
        # 假设 0.4mm nozzle, 像素宽度 0.4mm? 
        # 100mm = 250 pixels.
        # 根据 Bambu P1S, 0.4mm nozzle, horizontal detail is limited.
        # 建议 pixel size = nozzle size (0.4mm) or half (0.2mm).
        # 让 pixel resolution 由输出尺寸决定。
        # 这里 load_image 暂时只存原图，process 时再缩放
        self.image = img
        
    def process(self, settings: dict, materials: list[dict], 
                width_mm: float = 100, pixel_size_mm: float = 0.4,
                layer_height_mm: float = 0.08, layers: int = 5):
        """执行处理流程"""
        if self.image is None:
            return
            
        # 1. 计算目标像素尺寸
        h, w = self.image.shape[:2]
        # scale = width_mm / pixel_size_mm / max(h, w)
        target_w = int(width_mm / pixel_size_mm)
        target_h = int(target_w * h / w)
        
        # 缩放图像
        img_resized = cv2.resize(self.image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        current_img = img_resized
        
        # 2. 生成目标调色板 (Virtual Palette based on Materials)
        color_model = ColorModel(materials, layer_height=layer_height_mm, total_layers=layers)
        palette, combinations = color_model.generate_palette()
        self.palette = palette
        self.combinations = combinations
        
        # 3. 预处理 (Filter)
        filter_idx = settings.get('preprocess', 0)
        filter_algo = self.filters.get(filter_idx)
        if filter_algo:
            current_img = filter_algo.apply(current_img)
        
        # 获取距离度量设置
        distance_metric = settings.get('distance_metric', 'cie76')
        self._update_dither_settings(distance_metric)
        
        # 4. 检查是否使用矢量化模式
        vectorize_idx = settings.get('vectorize', 0)
        vectorizer = self.vectorizers.get(vectorize_idx)
        
        if vectorizer:
            # 使用矢量化处理 (跳过量化和抖动)
            processed_rgb, indices = vectorizer.apply(current_img, palette)
            self.processed = processed_rgb
            self.indices = indices
        else:
            # 传统处理流程: 量化 + 抖动
            
            # 4a. 色彩量化 (Reduction) - 可选
            quant_idx = settings.get('quantize', 0)
            quant_algo = self.quantizers.get(quant_idx)
            if quant_algo:
                current_img, _ = quant_algo.quantize(current_img, n_colors=64)
                
            # 4b. 抖动映射 (Dither to Target Palette)
            dither_idx = settings.get('dither', 0)
            dither_algo = self.dithers.get(dither_idx)
            
            if dither_algo:
                # 使用 Dither 算法得到视觉图像
                processed_rgb = dither_algo.apply(current_img, palette)
                self.processed = processed_rgb
                
                # 生成 Index Map (用于 3MF 生成)
                if distance_metric == 'cie76':
                    indices = self._match_colors_lab(processed_rgb, palette, target_h, target_w)
                else:
                    indices = self._match_colors_advanced(processed_rgb, palette, target_h, target_w, distance_metric)
                self.indices = indices
                
            else:
                # 无抖动，直接匹配
                if distance_metric == 'cie76':
                    indices = self._match_colors_lab(current_img, palette, target_h, target_w)
                else:
                    indices = self._match_colors_advanced(current_img, palette, target_h, target_w, distance_metric)
                    
                self.indices = indices
                self.processed = palette[self.indices]

    def _match_colors_lab(self, image: np.ndarray, palette: np.ndarray, h: int, w: int) -> np.ndarray:
        """使用 LAB 色彩空间进行颜色匹配"""
        from scipy.spatial import KDTree
        
        # 将 palette 转换为 LAB
        palette_rgb = palette.reshape(1, -1, 3).astype(np.uint8)
        palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        
        # 将图像转换为 LAB
        image_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # 在 LAB 空间构建 KDTree
        tree = KDTree(palette_lab)
        
        # 匹配
        flat_lab = image_lab.reshape(-1, 3)
        _, indices = tree.query(flat_lab)
        
        return indices.reshape(h, w)

    def _match_colors_advanced(self, image: np.ndarray, palette: np.ndarray, h: int, w: int, metric: str) -> np.ndarray:
        """
        使用高级颜色距离算法进行匹配 (Vectorized Hybrid Approach)
        1. 使用 KDTree 找到 K 个最近邻 (Euclidean LAB)
        2. 对 K 个候选者计算复杂距离 (CIEDE2000/CIE94)
        3. 选择最佳匹配
        """
        from scipy.spatial import KDTree
        from .color_distance import ciede2000_distance, cie94_distance, weighted_lab_distance
        
        # 准备数据
        palette_rgb = palette.reshape(1, -1, 3).astype(np.uint8)
        palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        
        image_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        flat_lab = image_lab.reshape(-1, 3)
        
        # [Optimization] 强制白色保护
        # 如果亮度 L > 98 (约250/255)，直接映射到最白的颜色
        # 找到 Palette 中最白的颜色索引 (最大 L)
        palette_L = palette_lab[:, 0]
        white_idx = np.argmax(palette_L)
        
        # 标记高亮像素
        L_channel = flat_lab[:, 0]
        # Calculate Chroma (Saturation) to avoid whitening colorful highlights (like cream)
        a_channel = flat_lab[:, 1]
        b_channel = flat_lab[:, 2]
        C_channel = np.sqrt(a_channel**2 + b_channel**2)
        
        # Condition: High Brightness AND Low Saturation
        # L > 230 (approx 90%)
        # C < 6 (very low saturation, almost neutral)
        high_key_mask = (L_channel > 230.0) & (C_channel < 6.0)
        
        is_high_key = high_key_mask
        
        # 1. 粗略筛选 (KDTree)
        k = 10 # 候选数量
        tree = KDTree(palette_lab)
        _, candidates_indices = tree.query(flat_lab, k=min(k, len(palette_lab))) # (N_pixels, K)
        
        # 2. 精确计算
        # 收集候选颜色的 LAB 值
        candidates_lab = palette_lab[candidates_indices] # (N, K, 3)
        
        # Flatten for vectorized distance function
        N = flat_lab.shape[0]
        K = candidates_indices.shape[1]
        
        p_flat = np.repeat(flat_lab, K, axis=0) # (N*K, 3)
        c_flat = candidates_lab.reshape(-1, 3)  # (N*K, 3)
        
        if metric == 'ciede2000':
            # Use kL=2.0 to be more tolerant of Lightness differences
            # This prioritizes Hue/Chroma matching, helping avoid "red/dark" artifacts
            # when the target is a light cream color that matches "White+Yellow" in Hue
            # but is slightly darker than the physical filament combination.
            dists_flat = ciede2000_distance(p_flat, c_flat, kL=2.0)
        elif metric == 'cie94':
            dists_flat = cie94_distance(p_flat, c_flat)
        elif metric == 'weighted':
            # Also adjust weighted distance to prioritize L less
            dists_flat = weighted_lab_distance(p_flat, c_flat, weights=(0.5, 1.0, 1.0))
        elif metric == 'oklab':
            # Use OKLab color space for better perceptual uniformity
            from .color_distance import rgb_to_oklab, oklab_distance
            # Convert from LAB back to RGB (approximately), then to OKLab
            # Better approach: work in OKLab from the start
            p_oklab = rgb_to_oklab(palette[candidates_indices.flatten()].reshape(-1, 3))
            img_oklab = rgb_to_oklab(image.astype(np.uint8))
            flat_oklab = img_oklab.reshape(-1, 3)
            p_flat_oklab = np.repeat(flat_oklab, K, axis=0)
            dists_flat = oklab_distance(p_flat_oklab, p_oklab)
        else:
            diff = p_flat - c_flat
            dists_flat = np.sum(diff**2, axis=1)
        
        # Reshape back to (N, K)
        dists = dists_flat.reshape(N, K)
        
        # 3. 选择最佳
        best_candidate_idx = np.argmin(dists, axis=1) # (N,) 0 to K-1
        
        # Map back to global palette index
        row_indices = np.arange(N)
        final_indices = candidates_indices[row_indices, best_candidate_idx]
        
        # 应用强制白色
        final_indices[is_high_key] = white_idx
        
        return final_indices.reshape(h, w)

    def get_layer_data(self) -> np.ndarray:
        """
        获取分层材料数据
        Returns: (H, W, Layers) 矩阵，值为 material_index
        """
        if self.indices is None or self.combinations is None:
            return None
            
        h, w = self.indices.shape
        layers = len(self.combinations[0])
        
        # 构造 (H, W, Layers) 矩阵
        # 这是一个巨大的矩阵，我们可能需要优化存储
        # 此时 self.combinations 是 list of tuples
        
        # 使用 numpy 广播
        # creating a lookup table from index to combination
        combo_lut = np.array(self.combinations, dtype=np.uint8) # (1024, 5)
        
        # self.indices is (H, W) values 0-1023
        # result (H, W, 5)
        layer_data = combo_lut[self.indices]
        return layer_data

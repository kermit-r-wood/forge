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
                layer_height_mm: float = 0.08, layers: int = 5,
                base_thickness_mm: float = 0.0):
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
        color_model = ColorModel(materials, layer_height=layer_height_mm, total_layers=layers, base_thickness=base_thickness_mm)
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
        
        # VTracer (index 2) 作为边缘平滑后处理，需要先抖动
        # ColorTraced (index 1) 仍作为独立矢量化器使用
        use_vtracer_smoothing = (vectorize_idx == 2 and vectorizer 
                                 and hasattr(vectorizer, 'smooth_edges')
                                 and vectorizer.is_available)
        use_standalone_vectorizer = (vectorizer and not use_vtracer_smoothing)
        
        if use_standalone_vectorizer:
            # ColorTraced 等独立矢量化器，跳过量化和抖动
            processed_rgb, indices = vectorizer.apply(current_img, palette)
            self.processed = processed_rgb
            self.indices = indices
        else:
            # 标准处理流程: 量化 + 抖动（+ 可选 VTracer 边缘平滑）
            
            # 4a. 色彩量化 (Reduction) - 可选
            quant_idx = settings.get('quantize', 0)
            quant_algo = self.quantizers.get(quant_idx)
            if quant_algo:
                current_img, _ = quant_algo.quantize(current_img, n_colors=64)
                
            # 4b. 抖动映射 (Dither to Target Palette)
            dither_idx = settings.get('dither', 0)
            dither_algo = self.dithers.get(dither_idx)
            
            if dither_algo:
                # 使用 Dither 算法得到抖动图像
                dithered_rgb = dither_algo.apply(current_img, palette)
                
                # 生成 Index Map (用于 3MF 生成)
                indices = self._match_colors_lab(dithered_rgb, palette, target_h, target_w)
                
                # Post-process: Clean up indices to avoid printer issues
                self.indices = self._clean_indices(indices, 
                                                 min_area=settings.get('min_area', 4),
                                                 kernel_size=settings.get('kernel_size', 1))
                
            else:
                # 无抖动，直接匹配
                indices = self._match_colors_lab(current_img, palette, target_h, target_w)
                
                # Post-process: Clean up indices
                self.indices = self._clean_indices(indices,
                                                 min_area=settings.get('min_area', 4),
                                                 kernel_size=settings.get('kernel_size', 1))
            
            # 4c. VTracer 边缘平滑后处理（保留抖动渐变，平滑区域边界）
            self.dithered_debug = self.indices.copy()
            if use_vtracer_smoothing:
                smoothed_rgb, smoothed_indices = vectorizer.smooth_edges(
                    img_resized, self.indices, palette)
                self.processed = smoothed_rgb
                self.indices = smoothed_indices
            else:
                # 预览使用 palette 颜色
                self.processed = self.palette[self.indices]

    def _clean_indices(self, indices: np.ndarray, min_area: int = 0, kernel_size: int = 1) -> np.ndarray:
        """
        Clean up indices map to remove noise and thin lines suitable for 3D printing.
        Optimized to prevent printer head clogging from small parts.
        """
        if indices is None:
            return None
            
        if min_area <= 1 and kernel_size <= 1:
            return indices
            
        h, w = indices.shape
        
        # Parameters - Adjusted to preserve dithering details while removing noise
        # Dithering produces small 1-2 pixel dots which are essential for the visual effect.
        # Too aggressive filtering destroys the image content.
        MIN_AREA = min_area
        KERNEL_SIZE = kernel_size
        kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
        kernel_close = np.ones((3, 3), np.uint8)  # Smaller closing kernel
        
        # Determine Background Index
        counts = np.bincount(indices.flatten())
        bg_index = np.argmax(counts)
        
        result_indices = indices.copy().astype(np.int32)
        unique_indices = np.unique(indices)
        
        for idx in unique_indices:
            if idx == bg_index:
                continue
                
            mask = (indices == idx).astype(np.uint8) * 255
            
            # A. Morphological Opening
            # Removes thin connections and small noise
            if KERNEL_SIZE > 1:
                mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            else:
                mask_opened = mask
            
            # B. Morphological Closing
            # Fills small holes and connects nearby regions to reduce fragmentation
            mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel_close)
            
            # C. Erosion to remove thin protrusions (followed by dilation to restore size)
            # This removes thin "fingers" that can cause printer head issues
            mask_eroded = cv2.erode(mask_closed, kernel, iterations=1)
            mask_cleaned = cv2.dilate(mask_eroded, kernel, iterations=1)
            
            # Identify pixels removed by filtering and set them to background
            pixels_removed = (mask > 0) & (mask_cleaned == 0)
            result_indices[pixels_removed] = bg_index
            
            # D. Area Filter
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_cleaned, connectivity=8)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < MIN_AREA:
                    result_indices[labels == i] = bg_index
                    
        return result_indices

    def _match_colors_lab(self, image: np.ndarray, palette: np.ndarray, h: int, w: int) -> np.ndarray:
        """使用 CIEDE2000 色彩距离进行颜色匹配"""
        from .color_distance import match_colors_ciede2000_numba
        
        # Convert RGB to float32 [0, 1] before LAB conversion to get standard LAB ranges
        palette_rgb = palette.reshape(1, -1, 3).astype(np.float32) / 255.0
        palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        
        # 将图像转换为 LAB
        image_rgb = image.astype(np.float32) / 255.0
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        flat_lab = image_lab.reshape(-1, 3)
        
        # 使用 Numba 优化的精确 CIEDE2000 距离多核匹配
        indices = match_colors_ciede2000_numba(flat_lab, palette_lab)
        
        return indices.reshape(h, w)


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


"""
VTracer 矢量化包装器
可选依赖，不可用时自动回退到 Color-Traced
SVG 渲染使用 Qt QSvgRenderer，确保正确处理所有 SVG 特性
"""
import io
import numpy as np
import cv2
from .base import BaseVectorizer
from .color_traced import ColorTracedVectorizer

try:
    import vtracer
    HAS_VTRACER = True
except ImportError:
    HAS_VTRACER = False


def _render_svg_to_numpy(svg_str: str, width: int, height: int) -> np.ndarray:
    """
    使用 Qt QSvgRenderer 将 SVG 字符串渲染为 RGB numpy 数组。
    Qt 正确处理所有 SVG 特性（transform, viewBox, 坐标系等）。
    渲染时关闭抗锯齿并使用 2x 超采样，避免边缘向白色混合导致过曝。
    """
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtCore import QByteArray, Qt

    # 2x 超采样渲染，减少多边形间微间隙
    RENDER_SCALE = 2
    render_w = width * RENDER_SCALE
    render_h = height * RENDER_SCALE

    svg_data = QByteArray(svg_str.encode('utf-8'))
    renderer = QSvgRenderer(svg_data)

    # 创建 QImage 作为渲染目标
    image = QImage(render_w, render_h, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)

    painter = QPainter(image)
    # 关闭抗锯齿，防止多边形边缘向白色背景混合（过曝的根因）
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    renderer.render(painter)
    painter.end()

    # QImage -> numpy array
    ptr = image.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(render_h, render_w, 4)
    # QImage Format_RGB32 是 BGRA 格式，提取 RGB
    rgb = arr[:, :, [2, 1, 0]].copy()
    # 最近邻缩回原始分辨率，保持锐利边缘
    if RENDER_SCALE != 1:
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_NEAREST)
    return rgb


class VTracerVectorizer(BaseVectorizer):
    """
    VTracer 矢量化包装器
    使用 Rust 编写的高性能矢量化库
    不可用时自动回退到 Color-Traced
    SVG 渲染使用 Qt QSvgRenderer
    """

    def __init__(self):
        self._fallback = ColorTracedVectorizer()

    @property
    def is_available(self) -> bool:
        return HAS_VTRACER

    def apply(self, image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """执行 VTracer 矢量化"""
        if not HAS_VTRACER:
            return self._fallback.apply(image, palette)

        h, w = image.shape[:2]

        # 1. 直接映射到调色板（确保输入到 vtracer 的颜色严格来自 palette）
        # 不做 K-Means 预量化，避免丢失 palette 中的颜色区分
        indices = self._map_to_palette(image, palette)
        mapped = palette[indices]

        # 3. vtracer 矢量化
        # 先将 palette 映射图放大 3x（最近邻），使 1px 细线变为 3px，
        # 避免 vtracer 追踪时丢弃过细的特征。SVG 是矢量格式，渲染回原始分辨率不受影响。
        UPSCALE = 3
        mapped_up = cv2.resize(mapped, (w * UPSCALE, h * UPSCALE),
                               interpolation=cv2.INTER_NEAREST)
        try:
            svg_str = self._vectorize(mapped_up)
            # 4. 用 Qt 渲染 SVG 回原始分辨率光栅图
            result = _render_svg_to_numpy(svg_str, w, h)
            # 5. 映射回 palette 索引
            indices = self._map_to_palette(result, palette)
            result = palette[indices]
        except Exception as e:
            print(f"[VTracer] 失败，回退到 ColorTraced: {e}")
            result, indices = self._fallback.apply(image, palette)

        return result, indices



    def _map_to_palette(self, image: np.ndarray, palette: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        from ..color_distance import match_colors_ciede2000_numba
        palette_rgb = palette.reshape(1, -1, 3).astype(np.float32) / 255.0
        palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        image_rgb = image.astype(np.float32) / 255.0
        image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        flat_lab = image_lab.reshape(-1, 3)
        indices = match_colors_ciede2000_numba(flat_lab, palette_lab)
        return indices.reshape(h, w)

    def _vectorize(self, image: np.ndarray) -> str:
        """使用 VTracer 将光栅图转为 SVG 字符串"""
        from PIL import Image

        pil_img = Image.fromarray(image)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()

        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format='PNG',
            colormode='color',
            hierarchical='stacked',
            mode='polygon',
            filter_speckle=0,
            color_precision=8,
            layer_difference=1,
            corner_threshold=15,
            length_threshold=0.5,
            max_iterations=15,
            splice_threshold=10,
            path_precision=8
        )
        return svg_str

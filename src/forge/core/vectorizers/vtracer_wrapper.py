"""
VTracer 矢量化包装器
可作为独立矢量化器使用，也可作为抖动结果的边缘平滑器。
SVG 渲染使用 Qt QSvgRenderer，确保正确处理所有 SVG 特性。
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
    关闭抗锯齿，避免边缘向白色背景混合。
    """
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtCore import QByteArray, Qt

    svg_data = QByteArray(svg_str.encode('utf-8'))
    renderer = QSvgRenderer(svg_data)

    image = QImage(width, height, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)

    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
    renderer.render(painter)
    painter.end()

    ptr = image.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(height, width, 4)
    rgb = arr[:, :, [2, 1, 0]].copy()
    return rgb


class VTracerVectorizer(BaseVectorizer):
    """
    VTracer 矢量化包装器
    支持两种模式:
    1. 独立矢量化 (apply) - 直接矢量化图像
    2. 边缘平滑 (smooth_edges) - 在抖动结果上平滑边缘
    """

    def __init__(self):
        self._fallback = ColorTracedVectorizer()

    @property
    def is_available(self) -> bool:
        return HAS_VTRACER

    def apply(self, image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """独立矢量化模式（向后兼容）"""
        if not HAS_VTRACER:
            return self._fallback.apply(image, palette)

        h, w = image.shape[:2]
        UPSCALE = 3
        image_up = cv2.resize(image, (w * UPSCALE, h * UPSCALE),
                              interpolation=cv2.INTER_NEAREST)
        try:
            svg_str = self._vectorize(image_up)
            result_up = _render_svg_to_numpy(svg_str, w * UPSCALE, h * UPSCALE)
            result = cv2.resize(result_up, (w, h), interpolation=cv2.INTER_NEAREST)
            indices = self._map_to_palette(result, palette)
            result = palette[indices]
        except Exception as e:
            print(f"[VTracer] 失败，回退到 ColorTraced: {e}")
            result, indices = self._fallback.apply(image, palette)

        return result, indices

    def smooth_edges(self, image: np.ndarray, dithered_indices: np.ndarray,
                     palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        边缘平滑模式：在抖动结果上用 VTracer 平滑颜色区域边界。
        内部保留抖动的像素图案（渐变细节），边缘使用 VTracer 的平滑多边形。

        关键：从 VTracer 的平滑区域检测边缘（真实颜色边界），
        而不是从抖动结果检测（抖动的每个交替像素都是"边界"）。

        Args:
            image: 原始图像 (H, W, 3) RGB
            dithered_indices: 抖动后的调色板索引图 (H, W)
            palette: 调色板 (N, 3) RGB

        Returns:
            tuple: (smoothed_rgb, smoothed_indices)
        """
        if not HAS_VTRACER:
            return palette[dithered_indices], dithered_indices

        h, w = image.shape[:2]
        UPSCALE = 3

        # 1. VTracer 矢量化原始图像，获取平滑的颜色区域
        image_up = cv2.resize(image, (w * UPSCALE, h * UPSCALE),
                              interpolation=cv2.INTER_NEAREST)
        try:
            svg_str = self._vectorize(image_up)
            result_up = _render_svg_to_numpy(svg_str, w * UPSCALE, h * UPSCALE)
            vt_result = cv2.resize(result_up, (w, h), interpolation=cv2.INTER_NEAREST)
            vt_indices = self._map_to_palette(vt_result, palette)
        except Exception as e:
            print(f"[VTracer] 边缘平滑失败: {e}")
            return palette[dithered_indices], dithered_indices

        # 2. 从 VTracer 的平滑区域检测边缘（真实颜色边界）
        #    VTracer 产生大块平滑多边形，边缘即真实颜色过渡处
        kernel = np.ones((3, 3), dtype=np.uint8)
        edge_mask = np.zeros((h, w), dtype=bool)

        for idx in np.unique(vt_indices):
            mask = (vt_indices == idx).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            eroded = cv2.erode(mask, kernel, iterations=1)
            edge_mask |= ((dilated - eroded) > 0)

        # 3. 合并：边缘用 VTracer（平滑边界），内部保留抖动（渐变细节）
        smoothed_indices = dithered_indices.copy()
        smoothed_indices[edge_mask] = vt_indices[edge_mask]

        smoothed_rgb = palette[smoothed_indices]
        return smoothed_rgb, smoothed_indices

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

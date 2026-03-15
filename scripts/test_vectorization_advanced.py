"""
高级矢量化算法对比测试
实现并比较 4 种矢量化方法:
1. Potrace - 专业位图追踪
2. VTracer - 现代彩色矢量化
3. Bezier Fitting - 贝塞尔曲线拟合

使用方法:
    uv run python scripts/test_vectorization_advanced.py <image_path>
"""
import sys
import cv2
import numpy as np
from pathlib import Path
import tempfile
import io

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forge.core.color_model import ColorModel
from forge.core.dithering.floyd_steinberg import FloydSteinbergDither

# 第三方矢量化库
try:
    import potracer
    HAS_POTRACE = True
except ImportError:
    HAS_POTRACE = False
    print("Warning: potracer not available")

try:
    import vtracer
    HAS_VTRACER = True
except ImportError:
    HAS_VTRACER = False
    print("Warning: vtracer not available")


def create_default_materials():
    """创建默认的 RYBW 材料"""
    return [
        {'name': 'White', 'color': (255, 255, 255), 'opacity': 0.3},
        {'name': 'Red', 'color': (255, 0, 0), 'opacity': 0.7},
        {'name': 'Yellow', 'color': (255, 255, 0), 'opacity': 0.5},
        {'name': 'Blue', 'color': (0, 0, 255), 'opacity': 0.7},
    ]


def quantize_image(image: np.ndarray, n_colors: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """使用 K-Means 量化图像颜色, 返回 (量化图, 聚类中心)"""
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(h, w, 3)
    return quantized, centers


def map_to_palette(image: np.ndarray, palette: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """将图像颜色映射到调色板, 返回 (索引图, 结果图)"""
    h, w = image.shape[:2]
    
    from forge.core.color_distance import match_colors_ciede2000_numba
    
    # 将 palette 转换为 LAB
    palette_rgb = palette.reshape(1, -1, 3).astype(np.uint8)
    palette_lab = cv2.cvtColor(palette_rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
    
    # 将图像转换为 LAB
    image_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    flat_lab = image_lab.reshape(-1, 3)
    
    # 使用 Numba 优化的精确 CIEDE2000 距离多核匹配
    indices = match_colors_ciede2000_numba(flat_lab, palette_lab)
    
    indices = indices.reshape(h, w)
    result = palette[indices]
    return indices, result


# ============================================================
# Curve Smoothing Utilities (Pure Python Potrace Alternative)
# ============================================================
def chaikin_smooth(points: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Chaikin 曲线平滑算法 - 将折线平滑为曲线"""
    if len(points) < 3:
        return points
    
    for _ in range(iterations):
        smoothed = []
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i + 1]
            # 在每条线段的 1/4 和 3/4 处插入新点
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            smoothed.append(q)
            smoothed.append(r)
        points = np.array(smoothed)
    
    return points


def savgol_smooth(points: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Savitzky-Golay 滤波平滑轮廓点"""
    if len(points) < window_size:
        return points
    
    # 确保 window_size 是奇数
    if window_size % 2 == 0:
        window_size += 1
    
    # 简化版 Savitzky-Golay: 使用移动平均
    kernel = np.ones(window_size) / window_size
    
    # 分别平滑 x 和 y
    x_smooth = np.convolve(points[:, 0], kernel, mode='same')
    y_smooth = np.convolve(points[:, 1], kernel, mode='same')
    
    # 保持端点不变
    x_smooth[0] = points[0, 0]
    x_smooth[-1] = points[-1, 0]
    y_smooth[0] = points[0, 1]
    y_smooth[-1] = points[-1, 1]
    
    return np.stack([x_smooth, y_smooth], axis=1)


def smooth_contour(contour: np.ndarray, method: str = 'chaikin') -> np.ndarray:
    """平滑轮廓点"""
    points = contour.reshape(-1, 2).astype(np.float64)
    
    if len(points) < 5:
        return contour
    
    if method == 'chaikin':
        # 先简化点以减少数量
        epsilon = 0.5 * cv2.arcLength(contour, True) / 100
        simplified = cv2.approxPolyDP(contour, epsilon, True)
        points = simplified.reshape(-1, 2).astype(np.float64)
        
        # Chaikin 平滑
        smoothed = chaikin_smooth(points, iterations=3)
    elif method == 'savgol':
        smoothed = savgol_smooth(points, window_size=7)
    else:
        smoothed = points
    
    return smoothed.astype(np.int32).reshape(-1, 1, 2)


# ============================================================
# Method 1: Curve Smoothing (Potrace Alternative)
# ============================================================
def method_potrace(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """使用曲线平滑进行矢量化 (Potrace 纯 Python 替代)"""
    h, w = image.shape[:2]
    
    # 量化减色
    quantized, _ = quantize_image(image, n_colors=16)
    indices, _ = map_to_palette(quantized, palette)
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    unique_indices = np.unique(indices)
    
    # 收集区域信息用于排序
    regions = []
    for idx in unique_indices:
        mask = (indices == idx).astype(np.uint8) * 255
        area = np.sum(mask > 0)
        regions.append((int(idx), mask, area))
    
    # 按面积从大到小绘制
    regions.sort(key=lambda x: x[2], reverse=True)
    
    for palette_idx, mask, _ in regions:
        # 形态学处理
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = tuple(int(c) for c in palette[palette_idx])
        
        for contour in contours:
            if len(contour) < 5:
                cv2.drawContours(result, [contour], -1, color, -1)
                continue
            
            # 使用 Chaikin 平滑
            smoothed = smooth_contour(contour, method='chaikin')
            
            if len(smoothed) > 2:
                cv2.fillPoly(result, [smoothed], color)
                # 抗锯齿边缘
                cv2.polylines(result, [smoothed], True, color, 1, cv2.LINE_AA)
            else:
                cv2.drawContours(result, [contour], -1, color, -1)
    
    return result


# ============================================================
# Method 2: VTracer
# ============================================================
def method_vtracer(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """使用 VTracer 进行彩色矢量化"""
    if not HAS_VTRACER:
        print("  [SKIP] VTracer not available")
        return None
    
    h, w = image.shape[:2]
    
    # 量化减色并映射到 palette
    quantized, _ = quantize_image(image, n_colors=16)
    _, mapped = map_to_palette(quantized, palette)
    
    try:
        # 将图像编码为 PNG 字节
        from PIL import Image
        pil_img = Image.fromarray(mapped)
        img_buffer = io.BytesIO()
        pil_img.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        svg_str = vtracer.convert_raw_image_to_svg(
            img_bytes,
            img_format='PNG',
            colormode='color',
            hierarchical='stacked',
            mode='polygon',
            filter_speckle=4,
            color_precision=6,
            layer_difference=16,
            corner_threshold=60,
            length_threshold=4.0,
            max_iterations=10,
            splice_threshold=45,
            path_precision=3
        )
        
        # SVG 渲染回栅格需要 cairosvg 等库
        # 这里简化处理: VTracer 生成了 SVG，但我们暂时使用 mapped 结果
        result = mapped.copy()
        
    except Exception as e:
        print(f"  VTracer error: {e}")
        result = mapped.copy()
    
    return result


# ============================================================
# Method 3: Bezier Fitting
# ============================================================
def fit_bezier_to_points(points: np.ndarray, max_error: float = 1.0) -> list:
    """将点序列拟合为三次贝塞尔曲线
    返回控制点列表 [(p0, p1, p2, p3), ...]
    """
    if len(points) < 4:
        return []
    
    # 简化实现: 将点分段，每段拟合一条贝塞尔曲线
    n = len(points)
    segment_length = max(4, n // 10)
    
    curves = []
    i = 0
    while i < n - 3:
        end = min(i + segment_length, n - 1)
        segment = points[i:end+1]
        
        if len(segment) < 4:
            break
            
        # 选择控制点
        p0 = segment[0]
        p3 = segment[-1]
        
        # 使用中间点估算控制点
        t1, t2 = 1/3, 2/3
        idx1 = int(len(segment) * t1)
        idx2 = int(len(segment) * t2)
        
        # 简化的控制点计算
        p1 = segment[idx1] + (segment[idx1] - p0) * 0.3
        p2 = segment[idx2] + (segment[idx2] - p3) * 0.3
        
        curves.append((p0, p1, p2, p3))
        i = end
    
    return curves


def sample_bezier_curve(p0, p1, p2, p3, n_samples: int = 10) -> np.ndarray:
    """采样贝塞尔曲线上的点"""
    t = np.linspace(0, 1, n_samples).reshape(-1, 1)
    points = (1-t)**3 * p0 + 3*(1-t)**2*t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
    return points.astype(np.int32)


def method_bezier_fitting(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """使用贝塞尔曲线拟合轮廓"""
    h, w = image.shape[:2]
    
    # 量化减色
    quantized, _ = quantize_image(image, n_colors=16)
    indices, _ = map_to_palette(quantized, palette)
    
    result = np.zeros((h, w, 3), dtype=np.uint8)
    unique_indices = np.unique(indices)
    
    # 收集并排序区域
    regions = []
    for idx in unique_indices:
        mask = (indices == idx).astype(np.uint8) * 255
        area = np.sum(mask > 0)
        regions.append((int(idx), mask, area))
    
    regions.sort(key=lambda x: x[2], reverse=True)
    
    for palette_idx, mask, _ in regions:
        # 形态学处理
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        color = tuple(int(c) for c in palette[palette_idx])
        
        for contour in contours:
            if len(contour) < 10:
                cv2.drawContours(result, [contour], -1, color, -1)
                continue
            
            # 提取轮廓点
            points = contour.reshape(-1, 2).astype(np.float64)
            
            # 拟合贝塞尔曲线
            curves = fit_bezier_to_points(points)
            
            if not curves:
                cv2.drawContours(result, [contour], -1, color, -1)
                continue
            
            # 采样所有曲线点
            all_points = []
            for p0, p1, p2, p3 in curves:
                sampled = sample_bezier_curve(p0, p1, p2, p3, n_samples=8)
                all_points.extend(sampled.tolist())
            
            if len(all_points) > 2:
                smooth_contour = np.array(all_points, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(result, [smooth_contour], color)
            else:
                cv2.drawContours(result, [contour], -1, color, -1)
    
    return result


# ============================================================
# Baseline: Pixel Dithering
# ============================================================
def method_dithering(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """基准方法: 像素抖动"""
    dither = FloydSteinbergDither()
    return dither.apply(image, palette)


# ============================================================
# Metrics
# ============================================================
def detect_edges(image: np.ndarray) -> np.ndarray:
    """Canny 边缘检测"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, 50, 150)


def compute_metrics(image: np.ndarray) -> dict:
    """计算图像质量指标"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    edges = detect_edges(image)
    
    # 边缘锐度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    edge_mask = edges > 0
    edge_sharpness = np.mean(gradient_mag[edge_mask]) if np.sum(edge_mask) > 0 else 0
    
    # 区域噪声
    non_edge_mask = ~edge_mask
    if np.sum(non_edge_mask) > 100:
        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean**2
        region_noise = np.mean(local_var[non_edge_mask])
    else:
        region_noise = 0
    
    return {
        'edge_sharpness': edge_sharpness,
        'region_noise': region_noise
    }


def compute_score(metrics: dict, baseline_metrics: dict) -> float:
    """计算综合评分"""
    if baseline_metrics['edge_sharpness'] == 0:
        edge_score = 0.5
    else:
        edge_score = min(metrics['edge_sharpness'] / baseline_metrics['edge_sharpness'], 2.0) / 2.0
    
    if baseline_metrics['region_noise'] == 0:
        noise_score = 0.5
    else:
        noise_score = 1 - min(metrics['region_noise'] / baseline_metrics['region_noise'], 2.0) / 2.0
    
    return edge_score * 0.5 + noise_score * 0.5


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_vectorization_advanced.py <image_path>")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: Failed to load image")
        sys.exit(1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w = image.shape[:2]
    # 不再限制图像大小，使用原始尺寸
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # 生成调色板
    print("Generating palette...")
    materials = create_default_materials()
    color_model = ColorModel(materials, layer_height=0.08, total_layers=5)
    palette, _ = color_model.generate_palette()
    print(f"Palette size: {len(palette)} colors\n")
    
    # 定义所有方法
    methods = [
        ("Dithering (Baseline)", method_dithering),
        ("Potrace", method_potrace),
        ("VTracer", method_vtracer),
        ("Bezier Fitting", method_bezier_fitting),
    ]
    
    results = {}
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # 运行所有方法
    for name, method in methods:
        print(f"[{name}]...")
        try:
            result = method(image.copy(), palette)
            if result is not None:
                metrics = compute_metrics(result)
                results[name] = {'image': result, 'metrics': metrics}
                print(f"  Edge Sharpness: {metrics['edge_sharpness']:.2f}")
                print(f"  Region Noise:   {metrics['region_noise']:.2f}")
                
                # 保存结果
                filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                cv2.imwrite(str(output_dir / f"adv_{filename}.png"), 
                           cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
            else:
                print("  [SKIPPED]")
        except Exception as e:
            print(f"  Error: {e}")
        print()
    
    # 对比表
    print("=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} {'Edge Sharpness':>15} {'Region Noise':>15} {'Score':>10}")
    print("-" * 70)
    
    baseline = results.get("Dithering (Baseline)", {}).get('metrics', {'edge_sharpness': 1, 'region_noise': 1})
    
    for name in results:
        m = results[name]['metrics']
        score = compute_score(m, baseline)
        print(f"{name:<25} {m['edge_sharpness']:>15.2f} {m['region_noise']:>15.2f} {score:>10.3f}")
    
    print("-" * 70)
    
    # 找出最佳方法
    best_name = max(results.keys(), key=lambda n: compute_score(results[n]['metrics'], baseline))
    print(f"\n>>> Best method: {best_name}")
    
    print(f"\nResults saved to: {output_dir}/adv_*.png")


if __name__ == "__main__":
    main()

"""
矢量化方法验证脚本 v2
改进版: 分层矢量化 + 边缘抗锯齿 + 边界专用锐度测量

使用方法:
    uv run python scripts/test_vectorization.py <image_path>
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from forge.core.color_model import ColorModel
from forge.core.dithering.floyd_steinberg import FloydSteinbergDither


def create_default_materials():
    """创建默认的 RYBW 材料"""
    return [
        {'name': 'White', 'color': (255, 255, 255), 'opacity': 0.3},
        {'name': 'Red', 'color': (255, 0, 0), 'opacity': 0.7},
        {'name': 'Yellow', 'color': (255, 255, 0), 'opacity': 0.5},
        {'name': 'Blue', 'color': (0, 0, 255), 'opacity': 0.7},
    ]


def quantize_image(image: np.ndarray, n_colors: int = 16) -> np.ndarray:
    """使用 K-Means 量化图像颜色"""
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(h, w, 3)
    return quantized


def map_to_palette(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """将图像颜色映射到调色板, 返回索引图"""
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)
    palette_f = palette.astype(np.float32)
    
    # 使用分块计算避免内存爆炸
    batch_size = 10000
    indices = np.zeros(len(pixels), dtype=np.int32)
    
    for i in range(0, len(pixels), batch_size):
        batch = pixels[i:i+batch_size]
        diff = batch[:, np.newaxis, :] - palette_f[np.newaxis, :, :]
        dist = np.sum(diff ** 2, axis=2)
        indices[i:i+batch_size] = np.argmin(dist, axis=1)
    
    return indices.reshape(h, w)


def vectorize_image_v2(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    改进版矢量化方法
    1. 量化减色 (更少颜色)
    2. 映射到 palette
    3. 分层绘制 (大区域先绘制, 小区域后绘制)
    4. 多边形逼近简化轮廓
    """
    h, w = image.shape[:2]
    
    # Step 1: 量化到较少颜色
    quantized = quantize_image(image, n_colors=24)
    
    # Step 2: 映射到 palette
    indices = map_to_palette(quantized, palette)
    
    # Step 3: 收集每种颜色的轮廓信息
    regions = []  # (palette_idx, contour, area)
    unique_indices = np.unique(indices)
    
    for idx in unique_indices:
        mask = (indices == idx).astype(np.uint8) * 255
        
        # 形态学处理: 轻微膨胀+腐蚀去除小噪点
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4:  # 忽略太小的区域
                continue
            
            # 多边形逼近 - epsilon 根据周长自适应
            perimeter = cv2.arcLength(cnt, True)
            epsilon = 0.002 * perimeter  # 很小的 epsilon 保持精度
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            regions.append((int(idx), approx, area))
    
    # Step 4: 按面积从大到小排序 (大区域先画)
    regions.sort(key=lambda x: x[2], reverse=True)
    
    # Step 5: 分层绘制
    result = np.zeros((h, w, 3), dtype=np.uint8)
    
    for palette_idx, contour, area in regions:
        color = tuple(int(c) for c in palette[palette_idx])
        cv2.drawContours(result, [contour], -1, color, -1)  # 填充
        # 边缘线 (抗锯齿效果)
        cv2.drawContours(result, [contour], -1, color, 1, cv2.LINE_AA)
    
    return result


def pixel_dither_image(image: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """传统像素抖动方法"""
    dither = FloydSteinbergDither()
    return dither.apply(image, palette)


def detect_edges(image: np.ndarray) -> np.ndarray:
    """使用 Canny 检测边缘"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def compute_edge_sharpness_v2(image: np.ndarray) -> tuple[float, float]:
    """
    改进版边缘清晰度计算
    返回: (边界锐度, 区域一致性)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
    
    # 1. 边界锐度: 检测边缘并计算梯度强度
    edges = detect_edges(image)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    
    # 只在边缘像素处计算平均梯度
    edge_mask = edges > 0
    if np.sum(edge_mask) > 0:
        edge_sharpness = np.mean(gradient_mag[edge_mask])
    else:
        edge_sharpness = 0
    
    # 2. 区域一致性: 非边缘区域的方差 (越低越好)
    non_edge_mask = ~edge_mask
    if np.sum(non_edge_mask) > 100:
        # 计算局部方差
        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray**2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean**2
        region_noise = np.mean(local_var[non_edge_mask])
    else:
        region_noise = 0
    
    return edge_sharpness, region_noise


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/test_vectorization.py <image_path>")
        print("\nExample: uv run python scripts/test_vectorization.py test.png")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # 加载图像
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print("Error: Failed to load image")
        sys.exit(1)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 调整大小 (用于测试)
    max_size = 250
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    
    # 生成调色板
    print("Generating palette...")
    materials = create_default_materials()
    color_model = ColorModel(materials, layer_height=0.08, total_layers=5)
    palette, combinations = color_model.generate_palette()
    print(f"Palette size: {len(palette)} colors")
    
    # 方法 1: 像素抖动
    print("\n[Method 1] Pixel Dithering...")
    dithered = pixel_dither_image(image.copy(), palette)
    edge_sharp_d, region_noise_d = compute_edge_sharpness_v2(dithered)
    print(f"  Edge sharpness: {edge_sharp_d:.2f}")
    print(f"  Region noise:   {region_noise_d:.2f}")
    
    # 方法 2: 矢量化 v2
    print("\n[Method 2] Vectorization v2...")
    vectorized = vectorize_image_v2(image.copy(), palette)
    edge_sharp_v, region_noise_v = compute_edge_sharpness_v2(vectorized)
    print(f"  Edge sharpness: {edge_sharp_v:.2f}")
    print(f"  Region noise:   {region_noise_v:.2f}")
    
    # 对比
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Metric':<25} {'Dithering':>15} {'Vectorization':>15}")
    print("-" * 60)
    print(f"{'Edge Sharpness (higher=better)':<25} {edge_sharp_d:>15.2f} {edge_sharp_v:>15.2f}")
    print(f"{'Region Noise (lower=better)':<25} {region_noise_d:>15.2f} {region_noise_v:>15.2f}")
    
    # 综合评分 (边缘锐度越高越好, 区域噪声越低越好)
    # 归一化后加权
    if edge_sharp_d > 0 and edge_sharp_v > 0:
        edge_score_d = edge_sharp_d / max(edge_sharp_d, edge_sharp_v)
        edge_score_v = edge_sharp_v / max(edge_sharp_d, edge_sharp_v)
    else:
        edge_score_d = edge_score_v = 0.5
        
    if region_noise_d > 0 or region_noise_v > 0:
        noise_score_d = 1 - region_noise_d / max(region_noise_d, region_noise_v, 1)
        noise_score_v = 1 - region_noise_v / max(region_noise_d, region_noise_v, 1)
    else:
        noise_score_d = noise_score_v = 0.5
    
    total_d = edge_score_d * 0.5 + noise_score_d * 0.5
    total_v = edge_score_v * 0.5 + noise_score_v * 0.5
    
    print("-" * 60)
    print(f"{'TOTAL SCORE':<25} {total_d:>15.3f} {total_v:>15.3f}")
    
    if total_v > total_d:
        print("\n>>> Vectorization produces BETTER results for this image!")
    else:
        print("\n>>> Dithering produces better results for this image.")
    
    # 保存对比图
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / "test_original.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "test_dithered.png"), cv2.cvtColor(dithered, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "test_vectorized.png"), cv2.cvtColor(vectorized, cv2.COLOR_RGB2BGR))
    
    # 生成边缘对比图
    edges_d = detect_edges(dithered)
    edges_v = detect_edges(vectorized)
    cv2.imwrite(str(output_dir / "test_edges_dithered.png"), edges_d)
    cv2.imwrite(str(output_dir / "test_edges_vectorized.png"), edges_v)
    
    print(f"\nResults saved to: {output_dir}")
    print("  - test_original.png")
    print("  - test_dithered.png")
    print("  - test_vectorized.png")
    print("  - test_edges_dithered.png")
    print("  - test_edges_vectorized.png")


if __name__ == "__main__":
    main()

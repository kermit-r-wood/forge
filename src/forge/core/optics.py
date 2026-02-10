"""
光学计算模块 - 改进版
使用基于物理的减色混合模型

Optical parameters can be calibrated using the CalibrationSolver.
"""
import numpy as np

# === Configurable Optical Model Parameters ===
# These can be optimized by the calibration solver to match actual print results.
# Default values are optimized for solid background (reflected light) viewing.

# Absorption factor: Controls color absorption strength (0.0 = no absorption, 1.0 = full Beer-Lambert)
ABSORPTION_FACTOR = 0.5

# Scatter contribution: Controls how much material color tints the light (0.0 = none, 1.0 = full)  
SCATTER_CONTRIBUTION = 0.5

# Scatter blend: Controls mixing ratio of transmitted vs scattered light (0.0 = all transmitted, 1.0 = all scattered)
SCATTER_BLEND = 0.15


def get_optical_params() -> dict:
    """Get current optical model parameters."""
    return {
        'absorption_factor': ABSORPTION_FACTOR,
        'scatter_contribution': SCATTER_CONTRIBUTION,
        'scatter_blend': SCATTER_BLEND
    }


def set_optical_params(absorption_factor: float = None, scatter_contribution: float = None, scatter_blend: float = None):
    """Set optical model parameters globally."""
    global ABSORPTION_FACTOR, SCATTER_CONTRIBUTION, SCATTER_BLEND
    if absorption_factor is not None:
        ABSORPTION_FACTOR = absorption_factor
    if scatter_contribution is not None:
        SCATTER_CONTRIBUTION = scatter_contribution
    if scatter_blend is not None:
        SCATTER_BLEND = scatter_blend


def parse_color(color) -> tuple:
    """解析颜色：支持十六进制字符串和 RGB 元组"""
    if isinstance(color, str):
        hex_col = color.lstrip('#')
        color_int = int(hex_col, 16)
        r = (color_int >> 16) & 0xFF
        g = (color_int >> 8) & 0xFF
        b = color_int & 0xFF
        return (r, g, b)
    return tuple(color)


def calculate_transmitted_color(
    layers: list[dict], 
    light_source: tuple = (255, 255, 255),
    absorption_factor: float = None,
    scatter_contribution: float = None,
    scatter_blend: float = None
) -> np.ndarray:
    """
    计算光线穿过多层半透明材料后的颜色
    
    使用减色混合模型：
    - 材料颜色代表该材料允许透过的光谱
    - 白色 (255,255,255) = 完全透明，不吸收任何光
    - 纯蓝 (0,0,255) = 吸收红色和绿色，只透过蓝色
    - 透明度(opacity)控制材料的散射/阻挡程度
    
    :param layers: 材料层列表
    :param light_source: 光源颜色 RGB (默认白光)
    :param absorption_factor: Absorption strength (None = use global default)
    :param scatter_contribution: Scatter color contribution (None = use global default)
    :param scatter_blend: Scatter/transmission blend ratio (None = use global default)
    :return: 透射光颜色 RGB (uint8)
    """
    # Use global defaults if not specified
    abs_factor = absorption_factor if absorption_factor is not None else ABSORPTION_FACTOR
    scat_contrib = scatter_contribution if scatter_contribution is not None else SCATTER_CONTRIBUTION
    scat_blend = scatter_blend if scatter_blend is not None else SCATTER_BLEND
    
    # 初始光强 (归一化到 0-1)
    current_light = np.array(light_source, dtype=np.float64) / 255.0
    
    ref_thickness = 0.08  # 参考层高 (mm)
    
    for layer in layers:
        # 解析颜色
        r, g, b = parse_color(layer['color'])
        material_color = np.array([r, g, b], dtype=np.float64) / 255.0
        
        opacity = layer['opacity']
        thickness = layer.get('thickness', ref_thickness)
        
        if thickness <= 0:
            continue
        
        # 计算有效散射率 (基于厚度调整)
        # opacity 代表标准厚度下的散射率
        scatter = 1.0 - (1.0 - opacity) ** (thickness / ref_thickness)
        
        # === 光学模型 ===
        # 透射分量：光线穿过材料时被选择性吸收
        # 吸收系数 = 1 - (material_color)，即材料颜色的互补色
        absorption_coeff = 1.0 - material_color
        
        # 应用 Beer-Lambert 定律 (指数衰减)
        # absorption_factor controls the absorption strength
        transmission = np.exp(-absorption_coeff * scatter * abs_factor)
        
        # 散射分量：部分光被材料散射并染上材料颜色
        # scatter_contribution controls the color tinting strength
        scattered_light = material_color * scatter * scat_contrib * np.mean(current_light)
        
        # 组合透射光和散射光
        # scatter_blend controls the mixing ratio
        transmitted_light = current_light * transmission
        current_light = transmitted_light * (1 - scatter * scat_blend) + scattered_light * scat_blend
        
        # 确保不超过 1
        current_light = np.clip(current_light, 0, 1)
    
    # 转换回 0-255 范围
    return np.clip(current_light * 255, 0, 255).astype(np.uint8)


def calculate_reflected_color(
    layers: list[dict], 
    light_source: tuple = (255, 255, 255),
    background: tuple = (255, 255, 255),
    absorption_factor: float = None,
    scatter_contribution: float = None
) -> np.ndarray:
    """
    Calculate the color seen when light reflects off stacked material layers.
    
    In reflected light mode (solid background viewing):
    - Light enters from the front (viewer side)
    - Passes through material layers, being absorbed
    - Reflects off the background (bottom layer / base)
    - Passes through layers again on the way back
    - Double-pass absorption results in more saturated colors
    
    :param layers: Material layers list (top to bottom, viewer to base)
    :param light_source: Incident light color RGB (default white)
    :param background: Background/base color RGB (default white)
    :param absorption_factor: Absorption strength (None = use global default)
    :param scatter_contribution: Scatter color contribution (None = use global default)
    :return: Reflected color RGB (uint8)
    """
    # Use global defaults if not specified
    abs_factor = absorption_factor if absorption_factor is not None else ABSORPTION_FACTOR
    scat_contrib = scatter_contribution if scatter_contribution is not None else SCATTER_CONTRIBUTION
    
    # Normalize to 0-1
    current_light = np.array(light_source, dtype=np.float64) / 255.0
    bg_color = np.array(background, dtype=np.float64) / 255.0
    
    ref_thickness = 0.08  # Reference layer height (mm)
    
    # Accumulated color from all layers (subtractive mixing)
    total_absorption = np.ones(3, dtype=np.float64)
    total_scatter = np.zeros(3, dtype=np.float64)
    
    for layer in layers:
        # Parse color
        r, g, b = parse_color(layer['color'])
        material_color = np.array([r, g, b], dtype=np.float64) / 255.0
        
        opacity = layer['opacity']
        thickness = layer.get('thickness', ref_thickness)
        
        if thickness <= 0:
            continue
        
        # Effective scatter rate based on thickness
        scatter = 1.0 - (1.0 - opacity) ** (thickness / ref_thickness)
        
        # Absorption coefficient = complement of material color
        # White material = no absorption, Blue material = absorbs R,G
        absorption_coeff = 1.0 - material_color
        
        # Single-pass absorption using Beer-Lambert
        single_pass = np.exp(-absorption_coeff * scatter * abs_factor)
        
        # Accumulate absorption (multiplicative)
        total_absorption *= single_pass
        
        # Accumulate scatter color contribution
        total_scatter += material_color * scatter * scat_contrib
    
    # Double-pass: light goes through layers twice (in and back out)
    # This is what makes reflected colors more saturated than transmitted
    double_pass_absorption = total_absorption ** 2
    
    # Light reflects off background
    reflected = current_light * double_pass_absorption * bg_color
    
    # Add scattered light contribution (tinting from materials)
    # Scattered light also undergoes double-pass
    scattered_contribution = total_scatter * np.mean(current_light) * 0.5
    
    # Combine reflected and scattered components
    final_color = reflected * 0.7 + scattered_contribution * 0.3
    
    # Ensure not exceeding 1
    final_color = np.clip(final_color, 0, 1)
    
    # Convert back to 0-255
    return np.clip(final_color * 255, 0, 255).astype(np.uint8)


def calculate_palette_preview(materials: list[dict], total_layers: int = 5, layer_height: float = 0.08) -> list:
    """
    预览所有可能的颜色组合（用于调试）
    返回代表性颜色样本
    """
    import itertools
    
    num_materials = len(materials)
    samples = []
    
    # 只采样部分组合（全组合太多）
    for combo in itertools.product(range(num_materials), repeat=total_layers):
        layers_data = []
        for mat_idx in combo:
            mat = materials[mat_idx]
            layers_data.append({
                'color': mat['color'],
                'opacity': mat['opacity'],
                'thickness': layer_height
            })
        
        color = calculate_transmitted_color(layers_data)
        samples.append({
            'combo': combo,
            'color': tuple(color.tolist())
        })
    
    return samples

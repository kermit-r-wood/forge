"""
光学计算模块 - 改进版
使用基于物理的减色混合模型

Optical parameters can be calibrated using the CalibrationSolver.
"""
import numpy as np

# === Configurable Optical Model Parameters ===
# These can be optimized by the calibration solver to match actual print results.
# Default values are optimized for solid background (reflected light) viewing.

# Absorption factor: Scales the K coefficient in K-M theory (higher = more saturated)
ABSORPTION_FACTOR = 1.0

# Scatter contribution: Scales the S coefficient in K-M theory (higher = lighter/whiter)
SCATTER_CONTRIBUTION = 0.15

# Scatter blend: Surface specular reflection factor (Fresnel-like, 0.0 = matte, higher = glossier)
SCATTER_BLEND = 0.02

# Absorption gamma: Power-law exponent for absorption mapping.
# Controls how partial absorption values (1-color) are transformed.
# gamma < 1: amplifies partial absorption, improves mixed-color fidelity
# gamma = 1: linear (original behavior)
# gamma > 1: suppresses partial absorption, more binary colors
ABSORPTION_GAMMA = 0.6


def get_optical_params() -> dict:
    """Get current optical model parameters."""
    return {
        'absorption_factor': ABSORPTION_FACTOR,
        'scatter_contribution': SCATTER_CONTRIBUTION,
        'scatter_blend': SCATTER_BLEND,
        'absorption_gamma': ABSORPTION_GAMMA
    }


def set_optical_params(absorption_factor: float = None, scatter_contribution: float = None,
                       scatter_blend: float = None, absorption_gamma: float = None):
    """Set optical model parameters globally."""
    global ABSORPTION_FACTOR, SCATTER_CONTRIBUTION, SCATTER_BLEND, ABSORPTION_GAMMA
    if absorption_factor is not None:
        ABSORPTION_FACTOR = absorption_factor
    if scatter_contribution is not None:
        SCATTER_CONTRIBUTION = scatter_contribution
    if scatter_blend is not None:
        SCATTER_BLEND = scatter_blend
    if absorption_gamma is not None:
        ABSORPTION_GAMMA = absorption_gamma


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


def _km_layer_RT(material_color: np.ndarray, opacity: float, thickness: float,
                  abs_factor: float, scat_contrib: float, gamma: float = 0.6,
                  ref_thickness: float = 0.08):
    """
    Compute per-channel Kubelka-Munk reflectance (R) and transmittance (T) for a single layer.
    
    K-M two-flux theory with power-law absorption:
      K_c = abs_factor * ((1 - color_c) ** gamma) * opacity / ref_thickness
      S   = scat_contrib * opacity / ref_thickness
    
    gamma < 1 amplifies partial absorption, improving mixed-color fidelity.
    Both K and S scale with opacity: transparent material (opacity=0) is invisible.
    
    For a layer of given thickness X:
      a = 1 + K/S,  b = sqrt(a^2 - 1)
      R = sinh(bSX) / (a*sinh(bSX) + b*cosh(bSX))
      T = b / (a*sinh(bSX) + b*cosh(bSX))
    """
    EPS = 1e-10
    
    # Per-channel absorption coefficient with power-law mapping
    # gamma < 1 amplifies weak absorption (e.g., 0.1^0.6 = 0.25 vs linear 0.1)
    absorption_raw = 1.0 - material_color  # complement of color
    absorption_mapped = np.power(np.maximum(absorption_raw, 0.0), gamma)
    K = abs_factor * absorption_mapped * opacity / ref_thickness  # shape (3,)
    
    # Achromatic scattering coefficient (per unit thickness)
    S_val = scat_contrib * opacity / ref_thickness  # scalar
    
    R = np.zeros(3, dtype=np.float64)
    T = np.ones(3, dtype=np.float64)
    
    if S_val < EPS:
        # No scattering: pure absorption, no reflection from this layer
        T = np.exp(-K * thickness)
        return R, T
    
    for c in range(3):
        k = K[c]
        a = 1.0 + k / S_val
        b_sq = a * a - 1.0
        
        if b_sq < EPS:
            # b ~ 0: K ~ 0 for this channel, pure scattering
            st = S_val * thickness
            R[c] = st / (1.0 + st)
            T[c] = 1.0 / (1.0 + st)
        else:
            b = np.sqrt(b_sq)
            arg = b * S_val * thickness
            
            if arg > 500:
                R[c] = a - b
                T[c] = 0.0
            else:
                sinh_arg = np.sinh(arg)
                cosh_arg = np.cosh(arg)
                denom = a * sinh_arg + b * cosh_arg
                
                if denom < EPS:
                    R[c] = 0.0
                    T[c] = 0.0
                else:
                    R[c] = sinh_arg / denom
                    T[c] = b / denom
    
    return R, T


def calculate_reflected_color(
    layers: list[dict], 
    light_source: tuple = (255, 255, 255),
    background: tuple = (255, 255, 255),
    absorption_factor: float = None,
    scatter_contribution: float = None,
    scatter_blend: float = None,
    absorption_gamma: float = None
) -> np.ndarray:
    """
    Calculate the color seen when light reflects off stacked material layers.
    
    Uses Kubelka-Munk two-flux theory with layer composition:
    - Each layer has per-channel reflectance R and transmittance T
    - Layers are composed bottom-up using K-M stacking formula
    - Power-law gamma controls absorption curve shape
    
    :param layers: Material layers list (top to bottom, viewer to base)
    :param light_source: Incident light color RGB (default white)
    :param background: Background/base color RGB (default white)
    :param absorption_factor: Absorption strength scaling K (None = use global default)
    :param scatter_contribution: Scattering strength scaling S (None = use global default)
    :param scatter_blend: Surface specular reflection factor (None = use global default)
    :param absorption_gamma: Power-law exponent for absorption (None = use global default)
    :return: Reflected color RGB (uint8)
    """
    # Use global defaults if not specified
    abs_factor = absorption_factor if absorption_factor is not None else ABSORPTION_FACTOR
    scat_contrib = scatter_contribution if scatter_contribution is not None else SCATTER_CONTRIBUTION
    scat_blend = scatter_blend if scatter_blend is not None else SCATTER_BLEND
    gamma = absorption_gamma if absorption_gamma is not None else ABSORPTION_GAMMA
    
    # Normalize to 0-1
    light = np.array(light_source, dtype=np.float64) / 255.0
    bg_color = np.array(background, dtype=np.float64) / 255.0
    
    ref_thickness = 0.08  # Reference layer height (mm)
    
    # Start with opaque background substrate
    R_stack = bg_color.copy()
    T_stack = np.zeros(3, dtype=np.float64)
    
    # Stack layers bottom-up: iterate in reverse (layers[0] = top/viewer side)
    for layer in reversed(layers):
        r, g, b = parse_color(layer['color'])
        material_color = np.array([r, g, b], dtype=np.float64) / 255.0
        
        opacity = layer['opacity']
        thickness = layer.get('thickness', ref_thickness)
        
        if thickness <= 0:
            continue
        
        # Compute K-M reflectance and transmittance for this layer
        R_layer, T_layer = _km_layer_RT(material_color, opacity, thickness,
                                         abs_factor, scat_contrib, gamma,
                                         ref_thickness)
        
        # K-M layer composition: add this layer on top of current stack
        denom = 1.0 - R_layer * R_stack
        denom = np.maximum(denom, 1e-10)
        
        R_new = R_layer + T_layer * T_layer * R_stack / denom
        T_new = T_layer * T_stack / denom
        
        R_stack = R_new
        T_stack = T_new
    
    # Body reflection: light enters material and reflects via K-M stack
    body = (1.0 - scat_blend) * light * R_stack
    
    # Surface specular: fraction of light reflects off surface without entering
    # This is additive (highlights) not multiplicative (desaturating)
    specular = scat_blend * light
    
    final_color = body + specular
    
    # Clamp and convert to 0-255
    final_color = np.clip(final_color, 0, 1)
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
        
        color = calculate_reflected_color(layers_data)
        samples.append({
            'combo': combo,
            'color': tuple(color.tolist())
        })
    
    return samples

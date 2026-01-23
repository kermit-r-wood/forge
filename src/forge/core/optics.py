"""
光学计算模块 - 改进版
使用基于物理的减色混合模型
"""
import numpy as np

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


def calculate_transmitted_color(layers: list[dict], light_source: tuple = (255, 255, 255)) -> np.ndarray:
    """
    计算光线穿过多层半透明材料后的颜色
    
    使用减色混合模型：
    - 材料颜色代表该材料允许透过的光谱
    - 白色 (255,255,255) = 完全透明，不吸收任何光
    - 纯蓝 (0,0,255) = 吸收红色和绿色，只透过蓝色
    - 透明度(opacity)控制材料的散射/阻挡程度
    
    :param layers: 材料层列表
    :param light_source: 光源颜色 RGB (默认白光)
    :return: 透射光颜色 RGB (uint8)
    """
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
        
        # === 改进的光学模型 ===
        # 
        # 当光线穿过材料时，发生两件事：
        # 1. 吸收 (Absorption): 材料吸收其互补色的光
        # 2. 散射 (Scattering): 部分光被散射，与材料颜色混合
        #
        # 透射系数 = 材料颜色 / 255 (白色=1表示完全透过)
        # 散射贡献 = 材料颜色 * 散射率 * 入射光强度
        
        # 透射分量：光线穿过材料时被选择性吸收
        # 吸收系数 = 1 - (material_color)，即材料颜色的互补色
        # 例如：蓝色材料 (0,0,1) 的吸收系数 = (1,1,0)，吸收红和绿
        absorption_coeff = 1.0 - material_color
        
        # 应用 Beer-Lambert 定律的简化形式
        # 透射率 = exp(-absorption * opacity * thickness_factor)
        # 使用近似：透射率 ≈ 1 - absorption * effective_opacity
        effective_absorption = absorption_coeff * scatter
        transmission = 1.0 - effective_absorption
        transmission = np.clip(transmission, 0, 1)
        
        # 散射分量：部分光被材料散射并染上材料颜色
        # 散射光强度与入射光和材料颜色成正比
        scattered_light = material_color * scatter * np.mean(current_light)
        
        # 组合透射光和散射光
        # 透射光 = 当前光 * 透射率
        # 最终输出 = 透射光 * (1 - scatter) + 散射光
        transmitted_light = current_light * transmission
        current_light = transmitted_light * (1 - scatter * 0.3) + scattered_light * 0.3
        
        # 确保不超过 1
        current_light = np.clip(current_light, 0, 1)
    
    # 转换回 0-255 范围
    return np.clip(current_light * 255, 0, 255).astype(np.uint8)


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

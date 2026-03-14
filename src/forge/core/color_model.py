"""
颜色模型模块
负责生成 RYBW 的所有可能组合 (Virtual Palette)
"""
import numpy as np
import itertools
from .optics import calculate_reflected_color

class ColorModel:
    """颜色模型"""
    
    def __init__(self, materials: list[dict], layer_height: float = 0.08, total_layers: int = 5, base_thickness: float = 0.0):
        """
        :param materials: 材料列表
        :param layer_height: 层高 (mm)
        :param total_layers: 总层数 (包括基底层)
        :param base_thickness: 底座厚度 (mm)
        """
        self.materials = materials
        self.layer_height = layer_height
        self.total_layers = total_layers
        self.base_thickness = base_thickness
        self.palette = None
        self.combinations = None # 记录每种颜色对应的材料层组合
    
    def _get_luma(self, color) -> float:
        """计算感知亮度"""
        if isinstance(color, str):
            h = color.lstrip('#')
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        elif isinstance(color, (list, tuple)):
            r, g, b = color
        else:
            return 0.0
        return r * 0.299 + g * 0.587 + b * 0.114
    
    def generate_palette(self) -> tuple[np.ndarray, list]:
        """
        生成所有可能的颜色组合
        :return: (palette_rgb_array, combinations_list)
        """
        # 生成所有层的材料索引组合
        # 假设最底层总是白色 (materials[0]) 用作基底，或者允许变化?
        # 通常 lithophane 底部是较厚的白色。
        # 这里假设所有 top layers (total_layers) 都可以是任意材料。
        # 4种材料 ^ 5层 = 1024 种组合
        
        num_materials = len(self.materials)
        # 生成层组合迭代器 (0, 0, 0, 0, 0) 到 (3, 3, 3, 3, 3)
        # indices 列表
        all_combinations = list(itertools.product(range(num_materials), repeat=self.total_layers))
        
        palette = []
        valid_combinations = []
        
        for combo in all_combinations:
            # 构建完整 layers 数据
            layers_data = []
            for mat_idx in combo:
                mat = self.materials[mat_idx]
                layers_data.append({
                    'color': mat['color'],
                    'opacity': mat['opacity'],
                    'thickness': self.layer_height
                })
            
            # 校正高亮浅色材料盖在深色底材上的虚假透色预测
            # 物理真实中，白色、黄色等高亮色由于遮盖力极强（含钛白粉等），
            # 覆盖在红、蓝等深色上时，会阻断底层透光。
            # K-M 公式会低估这种遮光特性，给出虚假的高饱和幻象色。
            # 校正方法：对这类组合，只用前 2 层做反射计算，忽略被物理遮挡的底层。
            needs_correction = False
            if len(set(combo)) > 1:  # 非纯色组合才需要检查
                combo_lumas = [self._get_luma(self.materials[mat_idx]['color']) for mat_idx in combo]
                top_luma = combo_lumas[0]
                
                if top_luma > 240:
                    # 纯白系顶层：物理上完全遮挡底层
                    needs_correction = True
                elif top_luma > 180:
                    # 高亮浅色（如黄色）顶层：如果底层有深色，物理上也会被遮挡
                    min_bottom_luma = min(combo_lumas[1:])
                    if min_bottom_luma < 150:
                        needs_correction = True
            
            if needs_correction:
                # 只用前 2 层计算反射色，模拟物理上表层遮挡底层的真实效果
                color = calculate_reflected_color(layers_data[:2], light_source=(255, 255, 255))
            else:
                # 正常全层 K-M 计算
                color = calculate_reflected_color(layers_data, light_source=(255, 255, 255))
            
            palette.append(color)
            valid_combinations.append(combo)
            
        self.palette = np.array(palette, dtype=np.uint8)
        self.combinations = valid_combinations
        
        return self.palette, self.combinations
        
    def get_layer_combination(self, color_index: int) -> tuple:
        """根据 palette index 获取层组合"""
        if self.combinations and 0 <= color_index < len(self.combinations):
            return self.combinations[color_index]
        return None

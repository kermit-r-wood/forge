"""
颜色模型模块
负责生成 RYBW 的所有可能组合 (Virtual Palette)
"""
import numpy as np
import itertools
from .optics import calculate_transmitted_color

class ColorModel:
    """颜色模型"""
    
    def __init__(self, materials: list[dict], layer_height: float = 0.08, total_layers: int = 5):
        """
        :param materials: 材料列表
        :param layer_height: 层高 (mm)
        :param total_layers: 总层数 (包括基底层)
        """
        self.materials = materials
        self.layer_height = layer_height
        self.total_layers = total_layers
        self.palette = None
        self.combinations = None # 记录每种颜色对应的材料层组合
    
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
            # 构建 layers 数据
            layers_data = []
            for mat_idx in combo:
                mat = self.materials[mat_idx]
                layers_data.append({
                    'color': mat['color'], # (R,G,B)
                    'opacity': mat['opacity'],
                    'thickness': self.layer_height
                })
            
            # 计算最终颜色
            # 假设有一个强白光背光
            color = calculate_transmitted_color(layers_data, light_source=(255, 255, 255))
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

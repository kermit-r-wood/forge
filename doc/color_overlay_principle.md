# 四色叠加生成多色原理

## 概述

Forge 主要使用 **RYBW（红、黄、蓝、白）** 或 **CMYK** 等透光 PLA 耗材，通过多层叠加实现全彩打印效果。这种方法基于**减色混合原理**和**光的透射特性**。支持用户自定义材料颜色和透光度。

---

## 物理原理

### 1. 减色混合 (Subtractive Color Mixing)

与屏幕显示的加色混合（RGB）不同，透明材料叠加遵循减色原理：

| 叠加组合 | 结果颜色 |
|----------|----------|
| 红 + 黄 | 橙色 |
| 黄 + 蓝 | 绿色 |
| 红 + 蓝 | 紫色 |
| 红 + 黄 + 蓝 | 棕/黑色 |
| 白（背光） | 提供光源 |

### 2. Beer-Lambert 定律

光线穿过半透明材料时，强度按指数衰减：

```
I = I₀ × e^(-α × d)
```

其中：
- `I₀` = 入射光强度
- `α` = 吸收系数（与材料颜色相关）
- `d` = 材料厚度
- `I` = 透射光强度

### 3. 透明度与吸收

每种颜色的材料选择性吸收特定波长：

| 材料颜色 | 吸收的光 | 透过的光 |
|----------|----------|----------|
| 红色 | 绿、蓝 | 红 |
| 黄色 | 蓝 | 红、绿 |
| 蓝色 | 红、绿 | 蓝 |
| 白色 | 无 | 全部 |

---

## 实现方法

### 层结构

```
观察者眼睛
    ↑
┌───────────┐ Layer 5 (顶层)
│  材料 X   │
├───────────┤ Layer 4
│  材料 Y   │
├───────────┤ Layer 3
│  材料 Z   │
├───────────┤ Layer 2
│  材料 W   │
├───────────┤ Layer 1 (底层)
│   白色    │ ← 通常用白色作为背光/底色
└───────────┘
    ↑
  光源（背光）
```

### 颜色组合计算

假设 4 种材料、5 层结构：
- 可能的颜色组合数 = 4^5 = **1024 种**
- 每种组合对应一个唯一的透射颜色

### 光学模拟算法

```python
def calculate_transmitted_color(layers, light_source):
    # 归一化光强 (0-1)
    current_light = light_source / 255.0
    
    for layer in layers:
        # 1. 计算透射分量 (Transmission)
        # 吸收系数 = 1 - (材料颜色), 即材料颜色的互补色
        absorption_coeff = 1.0 - layer.color_normalized
        
        # 计算有效散射率 (基于厚度)
        scatter = 1.0 - (1.0 - layer.opacity) ** (layer.thickness / ref_thickness)
        
        # 透射率 ≈ 1 - absorption * effective_opacity
        # 结合 Beer-Lambert 定律的简化
        effective_absorption = absorption_coeff * scatter
        transmission = 1.0 - effective_absorption
        transmission = clip(transmission, 0, 1)
        
        # 2. 计算散射分量 (Scattering)
        # 部分光被材料散射并染上材料颜色
        scattered_light = layer.color_normalized * scatter * mean(current_light)
        
        # 3. 组合透射光和散射光
        transmitted_light = current_light * transmission
        # 混合模型：透射光占主体，叠加部分散射光
        current_light = transmitted_light * (1 - scatter * 0.3) + scattered_light * 0.3
        
        current_light = clip(current_light, 0, 1)
    
    return current_light * 255
```

---

## 颜色匹配

### LAB 色彩空间

为了更准确地匹配颜色，Forge 使用 **LAB 色彩空间**而非 RGB：

- **L** = 亮度 (Lightness)
- **A** = 红绿轴 (Red-Green)
- **B** = 黄蓝轴 (Yellow-Blue)

LAB 空间的欧氏距离更接近人眼对颜色差异的感知。

### 匹配流程

```
原始像素 RGB → 转换为 LAB → 在调色板中找最近邻 → 返回层组合
```

---

## 抖动算法

由于调色板有限（1024 色），使用**误差扩散抖动**来提高视觉效果：

### Floyd-Steinberg 抖动

```
       X    7/16
 3/16  5/16 1/16
```

将量化误差分散到相邻像素，创造出更丰富的视觉色彩。

---

## 打印建议

| 参数 | 推荐值 |
|------|--------|
| 层高 | 0.08mm |
| 填充 | 100% |
| 颜色层数 | 5 层 |
| 底层材料 | 白色 PLA |
| 喷嘴 | 0.4mm |

---

## 参考资料

1. Beer-Lambert Law - [Wikipedia](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law)
2. Subtractive Color - [Wikipedia](https://en.wikipedia.org/wiki/Subtractive_color)
3. Floyd-Steinberg Dithering - [Wikipedia](https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering)
4. CIE LAB Color Space - [Wikipedia](https://en.wikipedia.org/wiki/CIELAB_color_space)

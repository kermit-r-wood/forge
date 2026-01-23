# Forge 项目设计文档

## 项目概述

**Forge** 是一个基于 RYBW（红、黄、蓝、白）四色叠加原理的多色 3MF 生成器，专为 Bambu Lab 等多色 3D 打印机设计。

### 核心功能

- 将 2D 图像转换为多色 3D 打印模型（3MF 格式）
- 支持多种图像预处理（Guided Filter 等）、量化和抖动算法（10+ 种）
- 基于光学原理模拟四色叠加效果，支持自定义材料
- 实时可视化算法效果对比与校准工具
- 跨平台支持 (Windows/Linux/macOS)
- 极致性能优化 (Vectorized Numpy Operations)

---

## 技术架构

```
forge/
├── src/forge/
│   ├── main.py              # 应用入口
│   ├── core/                # 核心算法
│   │   ├── analyzer.py      # 图像分析控制器
│   │   ├── calibration.py   # 校准矩阵生成
│   │   ├── color_model.py   # 颜色模型与调色板
│   │   ├── color_distance.py # 颜色距离计算 (CIEDE2000 等)
│   │   ├── optics.py        # 光学计算（改进的 Subtractive Mixing）
│   │   ├── exporter.py      # 3MF 导出器 (含面剔除优化)
│   │   ├── filters/         # 预处理滤波器 (Bilateral, Guided, Sharpen)
│   │   ├── quantizers/      # 色彩量化器
│   │   └── dithering/       # 抖动算法 (Floyd-Steinberg, Blue Noise, DBS 等)
│   └── ui/                  # 用户界面
│       ├── main_window.py   # 主窗口
│       ├── comparison_dialog.py  # 效果对比对话框
│       └── calibration_dialog.py # 校准工具对话框
└── doc/                     # 文档
```

---

## 模块说明

### 1. 核心处理流程 (Analyzer)

```
输入图像 → 预处理 → 量化 → 抖动 → 调色板映射 → 层数据生成
```

| 步骤 | 模块 | 说明 |
|------|------|------|
| 预处理 | `filters/` | Bilateral, Guided Filter, Sharpen, 平滑噪点保留边缘细节 |
| 量化 | `quantizers/` | K-Means、中值切割、八叉树，减少颜色数量 |
| 抖动 | `dithering/` | Floyd-Steinberg, Atkinson, Blue Noise, DBS, Sierra, Ordered 等多种算法 |
| 调色板 | `color_model.py` | 生成 4^n 种可能的颜色组合 (支持 1024 色) |

### 2. 光学模型 (Optics)

基于改进的减色混合模型 (Subtractive Mixing)，考虑吸收 (Absorption) 和散射 (Scattering)。
详见 [color_overlay_principle.md](./color_overlay_principle.md)

### 3. 颜色模型 (ColorModel)

- 输入：4 种材料定义（颜色 + 透光度）
- 输出：1024 种颜色的调色板（4 材料 × 5 层 = 4^5）
- 每种颜色对应一个层组合 (material_idx for each layer)

### 4. 3MF 导出 (Exporter)

- 使用 `lib3mf` 库 (实际实现使用 Native ZIP/XML 生成，移除对 lib3mf 依赖以提升便携性)
- **Vectorized Face Culling**: 使用 Numpy 矢量化逻辑剔除内部面，显著减少顶点数量
- 为每种材料生成独立的 MeshObject

---

## 依赖项

| 库 | 用途 |
|----|------|
| PySide6 | GUI 框架 |
| NumPy | 数值计算 |
| OpenCV | 图像处理 |
| scikit-learn | K-Means 聚类 |
| Pillow | 图像量化 |
| SciPy | KDTree 颜色匹配 |
| lib3mf | 3MF 文件生成 |

---

## 使用方法

```bash
# 安装依赖
uv sync

# 运行应用
uv run forge
```

---

## 开发路线

- [x] Phase 1: GUI 框架 + 基础布局
- [x] Phase 2: 多算法模块实现 (含高级抖动与滤波)
- [x] Phase 3: 材料配置系统
- [x] Phase 4: 3MF 导出 (Native XML 实现)
- [x] Phase 5: 性能优化（Vectorized Face Culling, Numpy 优化）
- [x] Phase 6: 跨平台打包与发布 (GitHub Actions, PyInstaller)
- [x] Phase 7: 校准与对比工具

---

## 许可证

MIT License

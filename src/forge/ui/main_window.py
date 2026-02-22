"""
Forge 主窗口
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox,
    QComboBox, QSlider, QProgressBar, QSplitter,
    QScrollArea, QFrame, QCheckBox, QDoubleSpinBox,
    QTabWidget, QStatusBar, QMenuBar, QMenu, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QSize, QThread
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor, QAction
from pathlib import Path
import numpy as np
import cv2
import ctypes
from ctypes import wintypes

from forge.core.analyzer import Analyzer
from forge.core.exporter import Exporter
from forge.core.settings import get_settings_manager
from forge.core.optics import set_optical_params

class SafeComboBox(QComboBox):
    """防止误触滚轮的 ComboBox"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class SafeDoubleSpinBox(QDoubleSpinBox):
    """防止误触滚轮的 DoubleSpinBox"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class SafeSlider(QSlider):
    """防止误触滚轮的 Slider"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class ImagePreviewWidget(QLabel):
    """图像预览组件"""
    
    def __init__(self, title: str = "预览"):
        super().__init__()
        self.title = title
        self.setMinimumSize(200, 200)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px dashed #555;
                border-radius: 8px;
                color: #888;
                font-size: 14px;
            }
        """)
        self.setText(title)
        self.setScaledContents(False)
        self._pixmap = None
    
    def set_image(self, image_path: str | Path):
        """设置预览图像 (从文件)"""
        pixmap = QPixmap(str(image_path))
        self._set_pixmap(pixmap)

    def set_image_array(self, image_array: np.ndarray):
        """设置预览图像 (从 numpy array)"""
        if image_array is None:
            return
            
        height, width, channel = image_array.shape
        bytes_per_line = 3 * width
        q_img = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self._set_pixmap(pixmap)
        
    def _set_pixmap(self, pixmap):
        if not pixmap.isNull():
            self._pixmap = pixmap
            scaled = pixmap.scaled(
                self.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)
    
    def resizeEvent(self, event):
        """调整大小时重新缩放图像"""
        super().resizeEvent(event)
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            super().setPixmap(scaled)


class AlgorithmPanel(QGroupBox):
    """算法选择面板"""
    
    def __init__(self):
        super().__init__("算法设置")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 预处理算法
        preprocess_group = QGroupBox("1. 预处理")
        preprocess_layout = QVBoxLayout(preprocess_group)
        self.preprocess_combo = SafeComboBox()
        self.preprocess_combo.addItems([
            "双边滤波",
            "引导滤波",
            "无预处理",
            "锐化"
        ])
        self.preprocess_combo.setCurrentIndex(2) # 默认无预处理
        preprocess_layout.addWidget(self.preprocess_combo)
        layout.addWidget(preprocess_group)
        
        # 量化算法
        quantize_group = QGroupBox("2. 色彩量化")
        quantize_layout = QVBoxLayout(quantize_group)
        self.quantize_combo = SafeComboBox()
        self.quantize_combo.addItems([
            "K-Means",
            "中值切割",
            "八叉树",
            "无量化 (原色)"
        ])
        self.quantize_combo.setCurrentIndex(3) # 默认无量化
        quantize_layout.addWidget(self.quantize_combo)
        layout.addWidget(quantize_group)
        
        # 抖动算法
        dither_group = QGroupBox("3. 抖动算法")
        dither_layout = QVBoxLayout(dither_group)
        self.dither_combo = SafeComboBox()
        self.dither_combo.addItems([
            "Floyd-Steinberg",          # 0
            "Atkinson",                 # 1
            "Sierra",                   # 2
            "无抖动",                   # 3
            "Blue Noise",               # 4
            "Ordered (Bayer)",          # 5
            "蛇形 FS (消除条纹)",        # 6
            "Hilbert 曲线",       # 7
            "结构感知 (保留边缘)",       # 8
            "DBS (极致画质/慢)"         # 9
        ])
        self.dither_combo.setCurrentIndex(1) # 默认 Atkinson
        dither_layout.addWidget(self.dither_combo)
        layout.addWidget(dither_group)
        
        # 细节优化 (Post-Process)
        opt_group = QGroupBox("4. 细节优化")
        opt_layout = QVBoxLayout(opt_group)
        
        # 杂点过滤 (Min Area)
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("杂点过滤:"))
        self.area_spin = SafeDoubleSpinBox() # Using DoubleSpinBox for consistency, but will store int
        self.area_spin.setDecimals(0)
        self.area_spin.setRange(0, 200)
        self.area_spin.setValue(0)
        self.area_spin.setToolTip("过滤掉小于此像素数的孤立区域。\n对于抖动模式建议设为 0-5 以保留细节。\n对于色块模式建议设为 20+ 以减少打印碎片。")
        area_layout.addWidget(self.area_spin)
        area_layout.addWidget(QLabel("px"))
        opt_layout.addLayout(area_layout)
        
        # 平滑力度 (Kernel Size)
        smooth_layout = QHBoxLayout()
        smooth_layout.addWidget(QLabel("平滑力度:"))
        self.smooth_spin = SafeDoubleSpinBox()
        self.smooth_spin.setDecimals(0)
        self.smooth_spin.setRange(1, 5)
        self.smooth_spin.setValue(1)
        self.smooth_spin.setSingleStep(2) # 1, 3, 5
        self.smooth_spin.setToolTip("形态学开运算核大小 (奇数)。\n1: 关闭 (保留所有细节)\n3: 轻微平滑\n5: 强力平滑")
        smooth_layout.addWidget(self.smooth_spin)
        opt_layout.addLayout(smooth_layout)
        
        layout.addWidget(opt_group)

        
        # 矢量化模式
        vectorize_group = QGroupBox("5. 矢量化模式")
        vectorize_layout = QVBoxLayout(vectorize_group)
        self.vectorize_combo = SafeComboBox()
        self.vectorize_combo.addItems([
            "关闭 (使用抖动)",         # 0
            "Color-Traced",    # 1
            "VTracer (可选)"           # 2
        ])
        self.vectorize_combo.setToolTip(
            "矢量化模式适用于动漫/插画/Logo 等扁平风格图像\n"
            "启用后将跳过量化和抖动，使用轮廓提取填充"
        )
        vectorize_layout.addWidget(self.vectorize_combo)
        layout.addWidget(vectorize_group)
        
        layout.addStretch()
    
    def get_settings(self) -> dict:
        """获取当前算法设置"""
        return {
            "preprocess": self.preprocess_combo.currentIndex(),
            "quantize": self.quantize_combo.currentIndex(),
            "dither": self.dither_combo.currentIndex(),
            "vectorize": self.vectorize_combo.currentIndex(),
            "min_area": int(self.area_spin.value()),
            "kernel_size": int(self.smooth_spin.value())
        }


class MaterialPanel(QGroupBox):
    """材料配置面板"""
    
    def __init__(self):
        super().__init__("材料配置")
        self._settings = get_settings_manager()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Scroll Area for materials
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")
        
        container = QWidget()
        self.materials_layout = QVBoxLayout(container)
        scroll.setWidget(container)
        
        layout.addWidget(scroll)
        
        # Load materials from settings
        saved_materials = self._settings.get_materials()
        
        self.material_widgets = []
        
        for mat in saved_materials:
            self._add_material_row(mat['name'], mat['color'], mat['opacity'])
            

        
    def _add_material_row(self, name, color, opacity):
        row = QFrame()
        row.setStyleSheet(".QFrame { background-color: #2d2d2d; border-radius: 4px; padding: 4px; margin-bottom: 2px; }")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(5, 5, 5, 5)
        
        # 颜色指示器
        color_label = QLabel()
        color_label.setFixedSize(24, 24)
        color_label.setStyleSheet(f"""
            background-color: {color};
            border: 1px solid #555;
            border-radius: 4px;
        """)
        row_layout.addWidget(color_label)
        
        # 信息布局
        info_layout = QVBoxLayout()
        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: bold;")
        info_layout.addWidget(name_label)
        
        # 透光度滑块
        slider_layout = QHBoxLayout()
        opacity_slider = SafeSlider(Qt.Horizontal)
        opacity_slider.setRange(0, 100)
        opacity_slider.setValue(int(opacity * 100))
        
        opacity_value = QLabel(f"{opacity:.2f}")
        opacity_value.setFixedWidth(35)
        
        slider_layout.addWidget(QLabel("透光:"))
        slider_layout.addWidget(opacity_slider)
        slider_layout.addWidget(opacity_value)
        info_layout.addLayout(slider_layout)
        
        row_layout.addLayout(info_layout, 1)
        
        # 更新数值显示并保存设置
        opacity_slider.valueChanged.connect(
            lambda v, lbl=opacity_value: lbl.setText(f"{v/100:.2f}")
        )
        opacity_slider.valueChanged.connect(self._save_materials)
        
        self.materials_layout.addWidget(row)
        
        self.material_widgets.append({
            "name": name,
            "color": color,
            "slider": opacity_slider
        })
        

    def _save_materials(self):
        """Save materials to settings file"""
        self._settings.set_materials(self.get_materials())
        
    def get_materials(self) -> list[dict]:
        """获取材料列表"""
        materials = []
        for w in self.material_widgets:
            materials.append({
                "name": w["name"],
                "color": w["color"], # Hex string
                "opacity": w["slider"].value() / 100.0
            })
        return materials

    def set_materials(self, materials_list: list[dict]):
        """设置材料列表 (清空并重新添加)"""
        # Clear layout
        # Remove items from layout backwards
        while self.materials_layout.count():
            item = self.materials_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.material_widgets = []
        
        for mat in materials_list:
            self._add_material_row(mat['name'], mat['color'], mat['opacity'])
        
        # Save to settings
        self._save_materials()


class OutputPanel(QGroupBox):
    """输出设置面板"""
    
    def __init__(self):
        super().__init__("输出设置")
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 输出尺寸
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("宽度 (mm):"))
        self.size_spin = SafeDoubleSpinBox()
        self.size_spin.setRange(10, 500)
        self.size_spin.setValue(200)
        size_layout.addWidget(self.size_spin)
        layout.addLayout(size_layout)
        
        # 像素大小 (同步 LD_ColorLayering 默认 0.6mm)
        pixel_layout = QHBoxLayout()
        pixel_layout.addWidget(QLabel("像素大小 (mm):"))
        self.pixel_spin = SafeDoubleSpinBox()
        self.pixel_spin.setRange(0.2, 1.0)
        self.pixel_spin.setValue(0.4)
        self.pixel_spin.setSingleStep(0.1)
        self.pixel_spin.setToolTip("每个像素对应的立方体边长。\nLD_ColorLayering 使用 0.6mm。")
        pixel_layout.addWidget(self.pixel_spin)
        layout.addLayout(pixel_layout)
        
        # 层高 (同步 LD_ColorLayering 默认 0.1mm)
        layer_layout = QHBoxLayout()
        layer_layout.addWidget(QLabel("层高 (mm):"))
        self.layer_spin = SafeDoubleSpinBox()
        self.layer_spin.setRange(0.04, 0.2)
        self.layer_spin.setValue(0.1)  # LD_ColorLayering 默认值
        self.layer_spin.setSingleStep(0.02)
        layer_layout.addWidget(self.layer_spin)
        layout.addLayout(layer_layout)
        
        # 总层数 (同步 LD_ColorLayering 默认 3 层)
        layers_layout = QHBoxLayout()
        layers_layout.addWidget(QLabel("颜色层数:"))
        self.layers_spin = SafeDoubleSpinBox()
        self.layers_spin.setDecimals(0)
        self.layers_spin.setRange(3, 8)
        self.layers_spin.setValue(4)
        self.layers_spin.setToolTip("颜色层数。")
        layers_layout.addWidget(self.layers_spin)
        layout.addLayout(layers_layout)
        
        # 底座厚度
        base_layout = QHBoxLayout()
        base_layout.addWidget(QLabel("底座厚度 (mm):"))
        self.base_spin = SafeDoubleSpinBox()
        self.base_spin.setRange(0.0, 2.0)
        self.base_spin.setValue(0.4)
        self.base_spin.setSingleStep(0.1)
        self.base_spin.setToolTip("实心底座层厚度，0 表示无底座。\n建议保持 0.4mm 以避免悬空区域和孔洞。")
        base_layout.addWidget(self.base_spin)
        layout.addLayout(base_layout)
        
        # 贪婪网格合并开关
        self.greedy_mesh_checkbox = QCheckBox("贪婪网格合并 (优化网格)")
        self.greedy_mesh_checkbox.setChecked(False)
        self.greedy_mesh_checkbox.setToolTip(
            "启用后将相邻相同材料像素合并为更大的矩形块。\n"
            "可减少网格复杂度，但可能影响打印效果。\n"
            "如果打印出现问题，请尝试关闭此选项。"
        )
        layout.addWidget(self.greedy_mesh_checkbox)
        
        layout.addStretch()
        
    def get_settings(self) -> dict:
        return {
            "width_mm": self.size_spin.value(),
            "pixel_size_mm": self.pixel_spin.value(),
            "layer_height_mm": self.layer_spin.value(),
            "layers": int(self.layers_spin.value()),
            "base_thickness_mm": self.base_spin.value(),
            "invert_z": True, # Default to Face Down
            "greedy_mesh": self.greedy_mesh_checkbox.isChecked()
        }


class ProcessingThread(QThread):
    """后台处理线程"""
    finished_signal = Signal(object) # 返回 processing result
    error_signal = Signal(str)
    
    def __init__(self, analyzer, settings, materials, output_settings):
        super().__init__()
        self.analyzer = analyzer
        self.settings = settings
        self.materials = materials
        self.output_settings = output_settings
        
    def run(self):
        try:
            self.analyzer.process(
                self.settings, 
                self.materials,
                width_mm=self.output_settings['width_mm'],
                pixel_size_mm=self.output_settings['pixel_size_mm'],
                layer_height_mm=self.output_settings['layer_height_mm'],
                layers=self.output_settings['layers'],
                base_thickness_mm=self.output_settings.get('base_thickness_mm', 0.0)
            )
            self.finished_signal.emit(self.analyzer.processed)
        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()


class ExportThread(QThread):
    """后台导出线程"""
    finished_signal = Signal()
    error_signal = Signal(str)
    
    def __init__(self, exporter, file_path, analyzer, materials, output_settings):
        super().__init__()
        self.exporter = exporter
        self.file_path = file_path
        self.analyzer = analyzer
        self.materials = materials
        self.output_settings = output_settings
        
    def run(self):
        try:
            layer_data = self.analyzer.get_layer_data()
            if layer_data is None:
                raise ValueError("无处理数据")
            
            # Get the processed RGB image for vertex colors
            rgb_image = self.analyzer.processed
                
            self.exporter.export(
                self.file_path,
                layer_data,
                self.materials,
                pixel_size_mm=self.output_settings.get('pixel_size_mm', 0.4),
                layer_height_mm=self.output_settings['layer_height_mm'],
                rgb_image=rgb_image,
                base_thickness_mm=self.output_settings.get('base_thickness_mm', 0.0),
                invert_z=self.output_settings.get('invert_z', False),
                greedy_mesh=self.output_settings.get('greedy_mesh', True)
            )
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))
            import traceback
            traceback.print_exc()


class MainWindow(QMainWindow):
    """Forge 主窗口"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Forge - 多色 3MF 生成器")
        self.setMinimumSize(1200, 800)
        
        self.analyzer = Analyzer()
        self.exporter = Exporter()
        self._current_image_path = None
        self._process_thread = None
        self._export_thread = None
        
        self._setup_menu()
        self._setup_ui()
        self._setup_statusbar()
        self._apply_dark_theme()
        self._set_windows_dark_title_bar()
        self._load_optical_params()

    def _set_windows_dark_title_bar(self):
        """设置 Windows 10/11 原生黑色标题栏"""
        try:
            # DWMWA_USE_IMMERSIVE_DARK_MODE = 20
            # DWMWA_MICA_EFFECT = 1029 (Windows 11 only)
            hwnd = int(self.winId())
            value = ctypes.c_int(1)
            
            # 尝试启用沉浸式暗黑模式 (Win10 1809+ / Win11)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 
                20, 
                ctypes.byref(value), 
                ctypes.sizeof(value)
            )
        except Exception:
            pass # Non-Windows or older version

    def _load_optical_params(self):
        """Load saved optical parameters from settings"""
        settings = get_settings_manager()
        saved_params = settings.get_optical_params()
        if saved_params:
            set_optical_params(**saved_params)

    def _setup_menu(self):
        """设置菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        open_action = QAction("打开图片(&O)", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("导出 3MF(&E)", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._export_3mf)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出(&X)", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具(&T)")
        
        calib_action = QAction("材料校准(&C)", self)
        calib_action.triggered.connect(self._show_calibration)
        tools_menu.addAction(calib_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助(&H)")
        about_action = QAction("关于(&A)", self)
        about_action.triggered.connect(lambda: QMessageBox.about(self, "关于 Forge", "Forge v0.1.0\n基于 RYBW 叠色原理的多色 3MF 生成器"))
        help_menu.addAction(about_action)
    
    def _setup_ui(self):
        """设置主界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # 左侧：图像预览区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # 原始图像预览
        self.original_preview = ImagePreviewWidget("原始图像")
        left_layout.addWidget(QLabel("原始图像"))
        left_layout.addWidget(self.original_preview, 1)
        
        # 颜色分离预览
        self.processed_preview = ImagePreviewWidget("处理结果")
        left_layout.addWidget(QLabel("处理结果 (Simulated)"))
        left_layout.addWidget(self.processed_preview, 1)
        
        # 右侧：设置面板
        right_panel = QWidget()
        right_panel.setFixedWidth(350)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # 使用 QScrollArea 替代 TabWidget
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        settings_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(0, 0, 5, 0) # Right padding for scrollbar
        settings_layout.setSpacing(15)
        
        # 算法面板
        self.algorithm_panel = AlgorithmPanel()
        settings_layout.addWidget(self.algorithm_panel)
        
        # 材料面板
        self.material_panel = MaterialPanel()
        self.material_panel.setMinimumHeight(250)
        settings_layout.addWidget(self.material_panel)
        
        # 输出面板
        self.output_panel = OutputPanel()
        settings_layout.addWidget(self.output_panel)
        
        settings_layout.addStretch()
        
        settings_scroll.setWidget(settings_container)
        right_layout.addWidget(settings_scroll)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("📂 导入图片")
        self.open_btn.setMinimumHeight(40)
        self.open_btn.clicked.connect(self._open_image)
        btn_layout.addWidget(self.open_btn)
        
        self.compare_btn = QPushButton("🔍 效果对比")
        self.compare_btn.setMinimumHeight(40)
        self.compare_btn.setEnabled(False)
        self.compare_btn.clicked.connect(self._show_comparison)
        self.compare_btn.setToolTip("并行计算多种算法组合，选择最佳效果")
        btn_layout.addWidget(self.compare_btn)
        
        right_layout.addLayout(btn_layout)
        
        # 第二行按钮
        btn_layout2 = QHBoxLayout()
        
        self.process_btn = QPushButton("🔄 处理图像")
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setEnabled(False)
        self.process_btn.clicked.connect(self._process_image)
        btn_layout2.addWidget(self.process_btn)
        
        self.export_btn = QPushButton("💾 导出 3MF")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_3mf)
        btn_layout2.addWidget(self.export_btn)
        
        right_layout.addLayout(btn_layout2)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        right_layout.addWidget(self.progress_bar)
        
        # 使用分割器
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        
        main_layout.addWidget(splitter)
        
    def closeEvent(self, event):
        """窗口关闭事件，清理线程"""
        if self._process_thread and self._process_thread.isRunning():
            self._process_thread.terminate()
            self._process_thread.wait()
        
        if self._export_thread and self._export_thread.isRunning():
            self._export_thread.terminate()
            self._export_thread.wait()
            
        super().closeEvent(event)
    
    def _setup_statusbar(self):
        """设置状态栏"""
        self.statusBar().showMessage("就绪")
    
    def _apply_dark_theme(self):
        """应用深色主题"""
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; font-family: "Segoe UI", sans-serif; }
            QGroupBox { font-weight: bold; border: 1px solid #3c3c3c; border-radius: 6px; margin-top: 12px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #0078d4; border: none; border-radius: 4px; padding: 8px 16px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #1e8ae6; }
            QPushButton:pressed { background-color: #006cbd; }
            QPushButton:disabled { background-color: #3c3c3c; color: #666; }
            QComboBox { background-color: #2d2d2d; border: 1px solid #3c3c3c; border-radius: 4px; padding: 5px; }
            QSlider::groove:horizontal { height: 6px; background: #3c3c3c; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; height: 16px; margin: -5px 0; background: #0078d4; border-radius: 8px; }
            QTabWidget::pane { border: 1px solid #3c3c3c; }
            QTabBar::tab { background-color: #2d2d2d; padding: 8px 16px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background-color: #1e1e1e; border-bottom: 2px solid #0078d4; }
            QSpinBox, QDoubleSpinBox { background-color: #2d2d2d; border: 1px solid #3c3c3c; padding: 5px; }
            QStatusBar { background-color: #007acc; color: white; }
            QMenuBar { 
                background-color: #1e1e1e; 
                border-bottom: 1px solid #3c3c3c;
                color: #e0e0e0;
            }
            QMenuBar::item {
                background-color: transparent;
                padding: 8px 12px;
                border-radius: 4px;
            }
            QMenuBar::item:selected { 
                background-color: #3c3c3c; 
            }
            QMenu { 
                background-color: #2d2d2d; 
                border: 1px solid #3c3c3c; 
                padding: 5px;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected { 
                background-color: #0078d4; 
                color: white;
            }
            QMenu::separator {
                height: 1px;
                background-color: #3c3c3c;
                margin: 5px 0;
            }
            QLabel { background-color: transparent; }
            QScrollArea { background-color: transparent; }
        """)
        
    def _open_image(self):
        """打开图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp *.gif);;所有文件 (*.*)"
        )
        if file_path:
            self._current_image_path = file_path
            self.original_preview.set_image(file_path)
            self.process_btn.setEnabled(True)
            self.compare_btn.setEnabled(True)  # 启用效果对比按钮
            try:
                self.analyzer.load_image(file_path)
                self.statusBar().showMessage(f"已加载: {Path(file_path).name}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法加载图像: {e}")
    
    def _show_comparison(self):
        """显示效果对比对话框"""
        if self.analyzer.image is None:
            return
        
        from forge.ui.comparison_dialog import ComparisonDialog
        
        materials = self.material_panel.get_materials()
        output_settings = self.output_panel.get_settings()
        algo_settings = self.algorithm_panel.get_settings()
        
        dialog = ComparisonDialog(
            self.analyzer.image,
            materials,
            output_settings['width_mm'],
            base_thickness_mm=output_settings.get('base_thickness_mm', 0.0),
            base_settings=algo_settings,
            parent=self
        )
        dialog.selection_made.connect(self._on_comparison_selected)
        dialog.exec()
    
    def _on_comparison_selected(self, settings):
        """用户从对比中选择了算法组合"""
        # 更新算法面板的选择
        self.algorithm_panel.preprocess_combo.setCurrentIndex(settings.get('preprocess', 0))
        self.algorithm_panel.quantize_combo.setCurrentIndex(settings.get('quantize', 0))
        self.algorithm_panel.dither_combo.setCurrentIndex(settings.get('dither', 0))
        
        # 更新矢量化模式
        self.algorithm_panel.vectorize_combo.setCurrentIndex(settings.get('vectorize', 0))
        
        # 更新细节优化参数
        if 'min_area' in settings:
            self.algorithm_panel.area_spin.setValue(settings['min_area'])
        if 'kernel_size' in settings:
            self.algorithm_panel.smooth_spin.setValue(settings['kernel_size'])

        
        # 自动执行处理
        self._process_image()
    
    
    def _process_image(self):
        """处理图像"""
        if not self._current_image_path:
            return
        
        self.statusBar().showMessage("正在处理图像...")
        self.progress_bar.setVisible(True)
        self.process_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        
        # 收集参数
        algo_settings = self.algorithm_panel.get_settings()
        materials = self.material_panel.get_materials()
        output_settings = self.output_panel.get_settings()
        
        # 启动线程
        self._process_thread = ProcessingThread(self.analyzer, algo_settings, materials, output_settings)
        self._process_thread.finished_signal.connect(self._on_process_finished)
        self._process_thread.error_signal.connect(self._on_process_error)
        self._process_thread.start()
        
    def _on_process_finished(self, processed_image):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        
        if processed_image is not None:
            self.processed_preview.set_image_array(processed_image)
            self.statusBar().showMessage("处理完成")
            
    def _on_process_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.process_btn.setEnabled(True)
        self.open_btn.setEnabled(True)
        QMessageBox.critical(self, "处理错误", f"图像处理失败: {error_msg}")
        self.statusBar().showMessage("处理失败")
    
    def _export_3mf(self):
        """导出 3MF 文件"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出 3MF",
            "output.3mf",
            "3MF 文件 (*.3mf)"
        )
        if file_path:
            self.statusBar().showMessage(f"正在导出: {Path(file_path).name}...")
            self.progress_bar.setVisible(True)
            self.export_btn.setEnabled(False)
            
            materials = self.material_panel.get_materials()
            
            # Use thread
            self._export_thread = ExportThread(
                self.exporter, file_path, self.analyzer, materials, self.output_panel.get_settings()
            )
            self._export_thread.finished_signal.connect(self._on_export_finished)
            self._export_thread.error_signal.connect(self._on_export_error)
            self._export_thread.start()
            
    def _on_export_finished(self):
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        self.statusBar().showMessage("导出完成")
        QMessageBox.information(self, "成功", "3MF 文件导出成功！")
        
    def _on_export_error(self, error_msg):
        self.progress_bar.setVisible(False)
        self.export_btn.setEnabled(True)
        QMessageBox.critical(self, "导出错误", f"导出失败: {error_msg}")
        self.statusBar().showMessage("导出失败")

    def _show_calibration(self):
        """显示材料校准对话框"""
        from forge.ui.calibration_dialog import CalibrationDialog
        
        current_materials = self.material_panel.get_materials()
        dialog = CalibrationDialog(current_materials, self)
        
        # 连接信号：当求解器计算出新参数时更新主界面
        dialog.materials_updated.connect(self._on_materials_calibrated)
        
        dialog.exec()

    def _on_materials_calibrated(self, new_materials):
        """应用校准后的材料参数"""
        self.material_panel.set_materials(new_materials)
        self.statusBar().showMessage("已应用新的材料校准参数")
        QMessageBox.information(self, "校准完成", "材料参数已更新！\n现在您可以重新处理图像以查看新效果。")

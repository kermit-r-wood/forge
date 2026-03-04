"""
算法比较对话框
并行计算多种算法组合，让用户选择最佳效果
"""
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QProgressBar, QScrollArea,
    QWidget, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread, QMutex
from PySide6.QtGui import QPixmap, QImage
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2


# 全局缓存 (按图片 hash 存储结果)
_results_cache = {}


def _get_image_hash(image: np.ndarray) -> str:
    """计算图像的简单 hash"""
    return str(hash(image.tobytes()[:10000]))  # 只用前 10KB 计算


class EnlargedPreviewDialog(QDialog):
    """放大预览对话框"""
    
    def __init__(self, image: np.ndarray, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setMinimumSize(600, 500)
        self.resize(800, 600)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 图像标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label)
        
        # 存储原始图像
        self._original_image = image.copy()
        self._update_image()
        
        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
            }
        """)
    
    def _update_image(self):
        """更新显示的图像"""
        if self._original_image is None:
            return
        
        # 获取可用空间
        available_size = self.image_label.size()
        if available_size.width() < 100 or available_size.height() < 100:
            available_size = self.size()
        
        h, w = self._original_image.shape[:2]
        scale = min(available_size.width() / w, available_size.height() / h, 2.0)  # 最大 2x
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(self._original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        h, w, ch = resized.shape
        q_img = QImage(resized.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_image()


class PreviewThumbnail(QFrame):
    """可点击的预览缩略图"""
    clicked = Signal(dict)  # 发送选中的设置
    double_clicked = Signal(np.ndarray, str)  # 发送图像和标题用于放大
    
    def __init__(self, settings: dict, label: str):
        super().__init__()
        self.settings = settings
        self.label = label
        self._image_data = None  # 存储原始图像数据
        
        self.setFrameStyle(QFrame.Box | QFrame.Plain)
        self.setLineWidth(2)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumSize(180, 140)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        
        # 图像预览
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a;")
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_label, 1)
        
        # 标签
        self.text_label = QLabel(label)
        self.text_label.setAlignment(Qt.AlignCenter)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("font-size: 10px; color: #aaa;")
        self.text_label.setMaximumHeight(40)
        layout.addWidget(self.text_label)
        
        self._selected = False
        self._update_style()
    
    def set_image(self, image: np.ndarray):
        """设置预览图像"""
        if image is None:
            return
        
        self._image_data = image.copy()
        self._refresh_thumbnail()
    
    def _refresh_thumbnail(self):
        """刷新缩略图显示"""
        if self._image_data is None:
            return
        
        # 获取可用大小
        available_size = self.image_label.size()
        if available_size.width() < 50 or available_size.height() < 50:
            available_size = self.size()
        
        h, w = self._image_data.shape[:2]
        max_w = max(available_size.width() - 10, 50)
        max_h = max(available_size.height() - 10, 50)
        scale = min(max_w / w, max_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        thumbnail = cv2.resize(self._image_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        h, w, ch = thumbnail.shape
        q_img = QImage(thumbnail.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_thumbnail()
    
    def set_selected(self, selected: bool):
        self._selected = selected
        self._update_style()
    
    def _update_style(self):
        if self._selected:
            self.setStyleSheet("""
                PreviewThumbnail {
                    background-color: #1e3a5f;
                    border: 2px solid #0078d4;
                    border-radius: 6px;
                }
            """)
        else:
            self.setStyleSheet("""
                PreviewThumbnail {
                    background-color: #2d2d2d;
                    border: 1px solid #3c3c3c;
                    border-radius: 6px;
                }
                PreviewThumbnail:hover {
                    border: 1px solid #0078d4;
                }
            """)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.settings)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_data is not None:
            self.double_clicked.emit(self._image_data, self.label)
        super().mouseDoubleClickEvent(event)


class ComparisonWorker(QThread):
    """后台并行处理多种算法组合"""
    progress = Signal(int, int)  # current, total
    result_ready = Signal(str, object)  # label, image
    finished_all = Signal()
    error = Signal(str)
    
    def __init__(self, image, materials, width_mm, combinations, base_thickness_mm=0.0, skip_labels=None):
        super().__init__()
        self.image = image.copy()
        self.materials = materials
        self.width_mm = width_mm
        self.base_thickness_mm = base_thickness_mm
        self.combinations = combinations
        self.skip_labels = skip_labels or set()
        self._mutex = QMutex()
        self._cancelled = False
    
    def cancel(self):
        self._mutex.lock()
        self._cancelled = True
        self._mutex.unlock()
        
    def run(self):
        # 过滤掉已缓存的组合
        to_process = [c for c in self.combinations if c["label"] not in self.skip_labels]
        
        total = len(to_process)
        if total == 0:
            self.finished_all.emit()
            return
        
        completed = 0
        
        # 使用线程池并发执行，加速多算法处理过程
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_combo = {
                executor.submit(self._process_single, combo): combo 
                for combo in to_process
            }
            
            for future in as_completed(future_to_combo):
                if self._check_cancelled():
                    break
                    
                combo = future_to_combo[future]
                try:
                    result = future.result()
                    if result is not None:
                        self.result_ready.emit(combo["label"], result)
                except Exception as e:
                    print(f"Error processing {combo}: {e}")
                
                completed += 1
                self.progress_bar_signal_hack(completed, total)
        
        self.finished_all.emit()
    
    def progress_bar_signal_hack(self, completed, total):
        # 避免在非主线程直接更新 UI
        self.progress.emit(completed, total)

    def _check_cancelled(self):
        self._mutex.lock()
        cancelled = self._cancelled
        self._mutex.unlock()
        return cancelled

    def _process_single(self, settings: dict) -> np.ndarray:
        """处理单个算法组合"""
        try:
            # Check cancellation inside the task
            if self._check_cancelled():
                return None
                
            from forge.core.analyzer import Analyzer
            analyzer = Analyzer()
            analyzer.image = self.image.copy()
            analyzer.process(settings, self.materials, width_mm=self.width_mm, base_thickness_mm=self.base_thickness_mm)
            if analyzer.processed is None:
                raise ValueError("Analyzer returned None result")
            return analyzer.processed
        except Exception as e:
            print(f"Process error for {settings.get('label')}: {e}")
            import traceback
            traceback.print_exc()
            # Return red error image
            h, w = self.image.shape[:2] # This is original size, need to respect resize?
            # Analyzer resizes internally. We should probably return None or handle scaling.
            # But PreviewThumbnail handles scaling.
            err_img = np.zeros((100, 100, 3), dtype=np.uint8)
            err_img[:, :] = [255, 0, 0] # Red
            return err_img


class ComparisonDialog(QDialog):
    """算法比较对话框"""
    
    COMBINATIONS = [
        # === 抖动算法对比 (固定: 无预处理 + 无量化) ===
        {"preprocess": 2, "quantize": 3, "dither": 0, "vectorize": 0, "label": "Q:无 | D:FS"},
        {"preprocess": 2, "quantize": 3, "dither": 1, "vectorize": 0, "label": "Q:无 | D:Atkinson"},
        {"preprocess": 2, "quantize": 3, "dither": 2, "vectorize": 0, "label": "Q:无 | D:Sierra"},
        {"preprocess": 2, "quantize": 3, "dither": 4, "vectorize": 0, "label": "Q:无 | D:BlueNoise"},
        {"preprocess": 2, "quantize": 3, "dither": 5, "vectorize": 0, "label": "Q:无 | D:Bayer"},
        {"preprocess": 2, "quantize": 3, "dither": 6, "vectorize": 0, "label": "Q:无 | D:蛇形FS"},
        {"preprocess": 2, "quantize": 3, "dither": 7, "vectorize": 0, "label": "Q:无 | D:Hilbert"},
        {"preprocess": 2, "quantize": 3, "dither": 8, "vectorize": 0, "label": "Q:无 | D:结构感知"},
        {"preprocess": 2, "quantize": 3, "dither": 9, "vectorize": 0, "label": "Q:无 | D:DBS"},
        {"preprocess": 2, "quantize": 3, "dither": 3, "vectorize": 0, "label": "Q:无 | D:无"},
    ]
    
    selection_made = Signal(dict)
    
    def __init__(self, image: np.ndarray, materials: list, width_mm: float, base_thickness_mm: float = 0.0, base_settings: dict = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("算法效果对比 (双击放大)")
        self.setWindowFlags(self.windowFlags() | Qt.WindowMinMaxButtonsHint | Qt.WindowCloseButtonHint)
        self.setMinimumSize(800, 600)
        self.resize(1000, 750)
        
        self.image = image
        self.materials = materials
        self.width_mm = width_mm
        self.base_thickness_mm = base_thickness_mm
        self.selected_settings = None
        self.thumbnails = {}
        self.worker = None
        self._image_hash = _get_image_hash(image)
        
        # Merge base_settings (min_area, kernel_size) into combinations
        self.combinations = []
        base = base_settings or {}
        min_area = base.get('min_area', 0) # Defaults to 0 if not provided
        kernel_size = base.get('kernel_size', 1)
        
        for combo in self.COMBINATIONS:
            c = combo.copy()
            c['min_area'] = min_area
            c['kernel_size'] = kernel_size
            self.combinations.append(c)
        
        self._setup_ui()
        self._load_cached_or_process()

    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        info = QLabel("⏳ 正在计算... 双击缩略图可放大查看")
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("font-size: 12px; color: #aaa; padding: 8px;")
        layout.addWidget(info)
        self.info_label = info
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(len(self.combinations))
        layout.addWidget(self.progress_bar)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        container = QWidget()
        self.grid_layout = QGridLayout(container)
        self.grid_layout.setSpacing(10)
        self.grid_layout.setContentsMargins(10, 10, 10, 10)
        
        for i, combo in enumerate(self.combinations):
            try:
                row, col = i // 3, i % 3
                thumbnail = PreviewThumbnail(combo, combo["label"])
                thumbnail.clicked.connect(self._on_thumbnail_clicked)
                thumbnail.double_clicked.connect(self._on_thumbnail_double_clicked)
                self.grid_layout.addWidget(thumbnail, row, col)
                self.thumbnails[combo["label"]] = thumbnail
            except Exception as e:
                print(f"Error creating thumbnail {i}: {e}")
        
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)
        
        btn_layout = QHBoxLayout()
        
        self.select_btn = QPushButton("使用选中效果")
        self.select_btn.setEnabled(False)
        self.select_btn.setMinimumHeight(36)
        self.select_btn.clicked.connect(self._confirm_selection)
        btn_layout.addWidget(self.select_btn)
        
        refresh_btn = QPushButton("🔄 重新对比")
        refresh_btn.setMinimumHeight(36)
        refresh_btn.clicked.connect(self._refresh_comparison)
        # 样式: 稍微暗一点的蓝色或灰色
        refresh_btn.setStyleSheet("""
            QPushButton { background-color: #2d2d2d; border: 1px solid #3c3c3c; }
            QPushButton:hover { background-color: #3c3c3c; border: 1px solid #555; }
        """)
        btn_layout.addWidget(refresh_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.setMinimumHeight(36)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; }
            QLabel { color: #e0e0e0; }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: white;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1e8ae6; }
            QPushButton:disabled { background-color: #3c3c3c; color: #666; }
            QProgressBar {
                border: none;
                background-color: #2d2d2d;
                border-radius: 4px;
                height: 6px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 4px;
            }
        """)
    
    def _load_cached_or_process(self):
        """加载缓存结果或开始处理"""
        global _results_cache
        
        cached = _results_cache.get(self._image_hash, {})
        cached_labels = set()
        
        # 恢复缓存的结果
        for label, image_data in cached.items():
            thumb = self.thumbnails.get(label)
            if thumb:
                thumb.set_image(image_data)
                cached_labels.add(label)
        
        # 计算需要处理的组合
        all_labels = {c["label"] for c in self.combinations}
        to_process = all_labels - cached_labels
        
        if not to_process:
            self.info_label.setText("✅ 已从缓存加载！双击放大，单击选择")
            self.progress_bar.setVisible(False)
            return
        
        # 开始处理未缓存的组合
        self.progress_bar.setMaximum(len(to_process))
        self.worker = ComparisonWorker(
            self.image,
            self.materials,
            self.width_mm,
            self.combinations,
            base_thickness_mm=self.base_thickness_mm,
            skip_labels=cached_labels
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result_ready)
        self.worker.finished_all.connect(self._on_finished)
        self.worker.start()
    
    def _on_progress(self, current, total):
        self.progress_bar.setValue(current)
    
    def _on_result_ready(self, label: str, image: np.ndarray):
        """单个结果就绪"""
        global _results_cache
        
        thumbnail = self.thumbnails.get(label)
        if thumbnail and image is not None:
            thumbnail.set_image(image)
            
            # 缓存结果
            if self._image_hash not in _results_cache:
                _results_cache[self._image_hash] = {}
            _results_cache[self._image_hash][label] = image.copy()
    
    def _on_finished(self):
        self.info_label.setText("✅ 处理完成！双击放大，单击选择")
        self.progress_bar.setVisible(False)
    
    def _on_thumbnail_clicked(self, settings):
        for thumb in self.thumbnails.values():
            thumb.set_selected(False)
        
        label = settings.get("label", "")
        thumbnail = self.thumbnails.get(label)
        if thumbnail:
            thumbnail.set_selected(True)
        
        self.selected_settings = settings
        self.select_btn.setEnabled(True)
    
    def _on_thumbnail_double_clicked(self, image: np.ndarray, title: str):
        """显示放大预览"""
        dialog = EnlargedPreviewDialog(image, title, self)
        dialog.exec()
    
    # 防止后台线程因 Python 对象回收而崩溃
    _detached_workers = []
    
    @classmethod
    def _keep_worker_alive(cls, worker):
        """保持 worker 存活直到结束"""
        cls._detached_workers.append(worker)
        # 当 worker 结束时，从列表中移除引用
        worker.finished.connect(lambda: cls._remove_worker(worker))
        worker.finished_all.connect(lambda: cls._remove_worker(worker))
        
    @classmethod
    def _remove_worker(cls, worker):
        if worker in cls._detached_workers:
            cls._detached_workers.remove(worker)
    
    def _refresh_comparison(self):
        """重新计算所有效果"""
        global _results_cache
        
        # 停止当前工作
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            # 将 worker 移交类级别管理，防止 GC 回收导致 crash
            self.worker.setParent(None)
            self._keep_worker_alive(self.worker)
            self.worker = None
        
        # 清除当前图片的缓存
        if self._image_hash in _results_cache:
            del _results_cache[self._image_hash]
        
        # 重置 UI
        for thumb in self.thumbnails.values():
            thumb.set_image(None) 
            thumb.image_label.setPixmap(QPixmap())
            thumb.image_label.setText("⏳")
            
        self.info_label.setText("⏳ 正在重新计算...")
        self.progress_bar.setVisible(True)
        self.select_btn.setEnabled(False)
        
        # 重新开始
        self._load_cached_or_process()

    def _confirm_selection(self):
        if self.selected_settings:
            self.selection_made.emit(self.selected_settings)
            self.accept()
    
    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            # 关键修复：防止 GC 回收 C++ 仍在运行的线程
            self.worker.setParent(None)
            self._keep_worker_alive(self.worker)
            self.worker = None
        super().closeEvent(event)
    
    def reject(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            # 关键修复：防止 GC 回收 C++ 仍在运行的线程
            self.worker.setParent(None)
            self._keep_worker_alive(self.worker)
            self.worker = None
        super().reject()

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTabWidget, QWidget, QTableWidget, QTableWidgetItem, 
    QHeaderView, QFileDialog, QMessageBox, QGroupBox, QColorDialog
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QImage, QPixmap, QColor
import numpy as np
from pathlib import Path

from forge.core.calibration import CalibrationGenerator, CalibrationSolver


class ClickableImageLabel(QLabel):
    clicked = Signal(int, int) # x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        super().setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.x(), event.y())

class CalibrationDialog(QDialog):
    materials_updated = Signal(list) # Emitted when solver updates materials

    def __init__(self, current_materials, parent=None):
        super().__init__(parent)
        self.setWindowTitle("材料校准工具")
        self.resize(1000, 750)
        self.current_materials = current_materials
        self._original_photo = None # Stores the loaded QImage
        self._setup_ui()
        self._load_preview()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        tabs = QTabWidget()
        
        # Tab 1: Generate Card
        card_tab = QWidget()
        self._setup_card_tab(card_tab)
        tabs.addTab(card_tab, "步骤1: 生成校准卡")
        
        # Tab 2: Solve parameters
        solve_tab = QWidget()
        self._setup_solve_tab(solve_tab)
        tabs.addTab(solve_tab, "步骤2: 计算参数")
        
        layout.addWidget(tabs)
        
    def _setup_card_tab(self, parent):
        layout = QHBoxLayout(parent)
        
        # Left: Preview
        preview_group = QGroupBox("校准卡预览")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #333; border: 1px solid #555;")
        self.preview_label.setMinimumSize(400, 400)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(preview_group, 2)
        
        # Right: Actions
        action_layout = QVBoxLayout()
        action_layout.addStretch()
        
        info_label = QLabel(
            "<h3>说明</h3>"
            "<p>1. 导出 3MF 文件并打印。</p>"
            "<p>2. 注意观察打印出的 16 个色块。</p>"
            "<p>3. 转到'步骤2'，输入你看到的实际颜色。</p>"
            "<p>此校准卡包含 CMY 单色梯度、混合灰阶及二次色。</p>"
        )
        info_label.setWordWrap(True)
        action_layout.addWidget(info_label)
        
        export_btn = QPushButton("💾 导出 3MF 文件")
        export_btn.setMinimumHeight(50)
        export_btn.clicked.connect(self._export_3mf)
        action_layout.addWidget(export_btn)
        
        save_img_btn = QPushButton("📷 保存预览图片")
        save_img_btn.clicked.connect(self._save_preview_image)
        action_layout.addWidget(save_img_btn)
        
        action_layout.addStretch()
        layout.addLayout(action_layout, 1)

    def _setup_solve_tab(self, parent):
        layout = QHBoxLayout(parent)
        
        # Left Panel: Picking Area
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("1. 导入照片 & 吸取颜色"))
        
        # Image Area
        self.pick_image_label = ClickableImageLabel()
        self.pick_image_label.setAlignment(Qt.AlignCenter)
        self.pick_image_label.setStyleSheet("background-color: #222; border: 1px solid #444;")
        self.pick_image_label.setMinimumSize(400, 400)
        self.pick_image_label.clicked.connect(self._on_image_clicked)
        left_layout.addWidget(self.pick_image_label, 1)
        
        # Import Button
        import_btn = QPushButton("📂 导入校准卡照片...")
        import_btn.setMinimumHeight(40)
        import_btn.clicked.connect(self._import_photo)
        left_layout.addWidget(import_btn)
        
        layout.addWidget(left_panel, 1)
        
        # Right Panel: Table & Solve
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("2. 确认颜色值 (先选表项，再点图片)"))
        
        # Table
        self.table = QTableWidget(4, 4)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.cellClicked.connect(self._on_table_cell_clicked) # Just track selection
        right_layout.addWidget(self.table)
        
        # Initialize table items
        self.observations = {} # (row, col) -> (r, g, b)
        for r in range(4):
            for c in range(4):
                item = QTableWidgetItem(f"色块 {r*4 + c + 1}")
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self.table.setItem(r, c, item)
                
        # Actions
        btn_layout = QHBoxLayout()
        solve_btn = QPushButton("🚀 开始计算")
        solve_btn.setMinimumHeight(40)
        solve_btn.clicked.connect(self._run_solver)
        btn_layout.addWidget(solve_btn)
        
        self.apply_btn = QPushButton("✅ 应用参数")
        self.apply_btn.setMinimumHeight(40)
        self.apply_btn.setEnabled(False)
        self.apply_btn.clicked.connect(self._apply_results)
        btn_layout.addWidget(self.apply_btn)
        
        right_layout.addLayout(btn_layout)
        
        # Console output
        self.result_label = QLabel("等待计算...")
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("padding: 10px; background-color: #222; font-family: monospace;")
        right_layout.addWidget(self.result_label)
        
        layout.addWidget(right_panel, 1)

    def _load_preview(self):
        patch_defs = CalibrationGenerator.get_16_color_patches()
        # Use current materials for preview
        img_arr = CalibrationGenerator.generate_preview(self.current_materials, patch_defs)
        
        # Convert to QPixmap
        h, w, ch = img_arr.shape
        img_q = QImage(img_arr.data, w, h, w*3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img_q)
        
        # Save as original (scaled up slightly for high quality)
        self._generated_preview_img = img_q
        
        # Display in Tab 1
        scaled = pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.preview_label.setPixmap(scaled)
        
        # Also display in Tab 2 by default (until user imports photo)
        self.pick_image_label.setPixmap(scaled)
        self._original_photo = img_q # Set default photo as the generated one
        
        # Colorize table
        for idx, counts in enumerate(patch_defs):
            if idx >= 16: break
            r = idx // 4
            c = idx % 4
            rgb = img_arr[r, c]
            color = QColor(rgb[0], rgb[1], rgb[2])
            item = self.table.item(r, c)
            item.setBackground(color)
            self.observations[(r,c)] = (rgb[0], rgb[1], rgb[2])
            
        # Select first cell by default
        self.table.setCurrentCell(0, 0)

    def _export_3mf(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "导出 3MF", "calibration.3mf", "3MF Files (*.3mf)")
        if file_path:
            try:
                patch_defs = CalibrationGenerator.get_16_color_patches()
                CalibrationGenerator.export_3mf(file_path, self.current_materials, patch_defs)
                QMessageBox.information(self, "成功", "导出成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"导出失败: {e}")

    def _save_preview_image(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "保存图片", "calibration_preview.png", "Images (*.png)")
        if file_path:
            self.preview_label.pixmap().save(file_path)

    def _import_photo(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择照片",
            "",
            "图片文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if file_path:
            image = QImage(file_path)
            if not image.isNull():
                self._original_photo = image
                
                # Show in label
                pixmap = QPixmap.fromImage(image)
                # Ensure it fits
                w = self.pick_image_label.width()
                h = self.pick_image_label.height()
                scaled = pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.pick_image_label.setPixmap(scaled)
                
                QMessageBox.information(self, "提示", "照片已加载。\n\n现在的操作流程：\n1. 在右侧表格选中一个单元格（例如 '色块1'）\n2. 在左侧照片上点击对应的位置吸取颜色。")

    def _on_table_cell_clicked(self, row, col):
        # Optional: Highlight target area on photo if we have AI segmentation? (Too advanced)
        # Just valid selection
        pass

    def _on_image_clicked(self, x, y):
        if self._original_photo is None:
            return
            
        # Need to map widget coordinate (x, y) back to image coordinate
        pixmap = self.pick_image_label.pixmap()
        if not pixmap: 
            return
            
        # The pixmap might be centered in the label
        # Get actual rect data
        # Alignment is Center
        # simplified mapping:
        
        # Label size
        lw = self.pick_image_label.width()
        lh = self.pick_image_label.height()
        
        # Pixmap size (displayed)
        pw = pixmap.width()
        ph = pixmap.height()
        
        # Top-left offset of the pixmap within label
        off_x = (lw - pw) // 2
        off_y = (lh - ph) // 2
        
        # Coordinate inside pixmap
        px_x = x - off_x
        px_y = y - off_y
        
        if px_x < 0 or px_x >= pw or px_y < 0 or px_y >= ph:
            return # Clicked outside image
            
        # Map to original image
        orig_w = self._original_photo.width()
        orig_h = self._original_photo.height()
        
        orig_x = int(px_x * orig_w / pw)
        orig_y = int(px_y * orig_h / ph)
        
        # Get Color
        pixel_color = self._original_photo.pixelColor(orig_x, orig_y)
        rgb = (pixel_color.red(), pixel_color.green(), pixel_color.blue())
        
        # Update currently selected table cell
        current_row = self.table.currentRow()
        current_col = self.table.currentColumn()
        
        if current_row < 0 or current_col < 0:
            QMessageBox.warning(self, "未选择", "请先在右侧表格中点击选择你要录入的色块。")
            return
            
        # Update Data
        self.observations[(current_row, current_col)] = rgb
        
        # Update Table GUI
        item = self.table.item(current_row, current_col)
        item.setBackground(pixel_color)
        item.setText(f"R{rgb[0]}\n{rgb[1]}\n{rgb[2]}")
        
    def _pick_color(self, row, col):
        # kept for legacy manual picking if user double clicks or something?
        # But we overrode cellClicked mainly for selection.
        # Let's keep manual override via context menu or double click?
        # For simple UX, let's keep cellClicked as Selection Only.
        # Maybe allow Double Click to open color dialog.
        pass

    def _run_solver(self):
        self.result_label.setText("正在计算 (可能需要几秒钟)...")
        # Collect observations list
        obs_list = []
        for (r, c), rgb in self.observations.items():
            idx = r * 4 + c
            obs_list.append((idx, rgb))
            
        # Run in thread ideally, but for now blocking is okay-ish for prototype (minimizer is fast for 16 points)
        try:
            optimized = CalibrationSolver.solve(self.current_materials, obs_list)
            self.optimized_materials = optimized
            
            # Show results
            text = "计算成功! 参数:\n"
            for m in optimized:
                text += f"{m['name']}: Opacity={m['opacity']:.2f}, Color={m['color']}\n"
            self.result_label.setText(text)
            self.apply_btn.setEnabled(True)
            
        except Exception as e:
            self.result_label.setText(f"计算失败: {e}")
            import traceback
            traceback.print_exc()

    def _apply_results(self):
        if hasattr(self, 'optimized_materials'):
            reply = QMessageBox.question(
                self, 
                "应用参数", 
                "确定要用这些新参数覆盖主界面的材料设置吗?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.materials_updated.emit(self.optimized_materials)
                self.close()

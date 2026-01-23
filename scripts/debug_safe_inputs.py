
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QComboBox, QDoubleSpinBox, QSlider, QScrollArea
from PySide6.QtCore import Qt

# Copy of Safe classes from main_window.py
class SafeComboBox(QComboBox):
    """防止误触滚轮的 ComboBox"""
    def wheelEvent(self, event):
        print(f"ComboBox WheelEvent. HasFocus: {self.hasFocus()}")
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class SafeDoubleSpinBox(QDoubleSpinBox):
    """防止误触滚轮的 DoubleSpinBox"""
    def wheelEvent(self, event):
        print(f"SpinBox WheelEvent. HasFocus: {self.hasFocus()}")
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class SafeSlider(QSlider):
    """防止误触滚轮的 Slider"""
    def wheelEvent(self, event):
        print(f"Slider WheelEvent. HasFocus: {self.hasFocus()}")
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

class DebugWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Debug Safe Inputs")
        self.resize(400, 600)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Scroll Area to test propagation
        scroll = QScrollArea()
        layout.addWidget(scroll)
        
        container = QWidget()
        scroll.setWidget(container)
        scroll.setWidgetResizable(True)
        
        vbox = QVBoxLayout(container)
        
        vbox.addWidget(QLabel("Try scrolling over these inputs WITHOUT clicking them first."))
        
        # Combo
        vbox.addWidget(QLabel("Safe ComboBox:"))
        cb = SafeComboBox()
        cb.addItems(["Item 1", "Item 2", "Item 3"])
        vbox.addWidget(cb)
        
        # Spin
        vbox.addWidget(QLabel("Safe DoubleSpinBox:"))
        sb = SafeDoubleSpinBox()
        vbox.addWidget(sb)
        
        # Slider
        vbox.addWidget(QLabel("Safe Slider:"))
        sl = SafeSlider(Qt.Horizontal)
        vbox.addWidget(sl)
        
        # Add spacing to allow scrolling
        vbox.addWidget(QLabel("Spacer 1"))
        vbox.addWidget(QLabel("Spacer 2"))
        vbox.addWidget(QLabel("Spacer 3"))
        for i in range(20):
            vbox.addWidget(QLabel(f"Spacer {i}"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = DebugWindow()
    win.show()
    sys.exit(app.exec())

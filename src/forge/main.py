"""
Forge 主应用程序入口
"""

import sys
from PySide6.QtWidgets import QApplication
from forge.ui.main_window import MainWindow


def main():
    """应用程序入口点"""
    app = QApplication(sys.argv)
    app.setApplicationName("Forge")
    app.setApplicationVersion("0.1.0")
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Forge 主应用程序入口
"""

import sys
import os

# PyInstaller Windows fix for vtracer (Rust panic on stdout/stderr)
if sys.platform == 'win32':
    try:
        null_fd = os.open(os.devnull, os.O_RDWR)
        for fd in (1, 2):
            try:
                os.dup2(null_fd, fd)
            except OSError:
                pass
    except Exception:
        pass

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

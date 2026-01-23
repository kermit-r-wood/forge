import sys
from PySide6.QtWidgets import QApplication
try:
    from forge.ui.main_window import MainWindow
    print("MainWindow imported successfully")
except Exception as e:
    print(f"Error importing MainWindow: {e}")
    import traceback
    traceback.print_exc()

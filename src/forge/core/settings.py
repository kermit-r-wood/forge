"""
Settings Manager - 配置存储模块

Handles persistent storage of application settings including materials configuration.
Uses JSON file in AppData for normal installation or portable mode for packaged builds.
"""
import json
import os
from pathlib import Path


class SettingsManager:
    """管理应用程序设置的持久化存储"""
    
    DEFAULT_MATERIALS = [
        {"name": "白色", "color": "#FFFFFF", "opacity": 0.3},
        {"name": "红色", "color": "#FF0000", "opacity": 0.7},
        {"name": "黄色", "color": "#FFFF00", "opacity": 0.6},
        {"name": "蓝色", "color": "#0000FF", "opacity": 0.7},
    ]
    
    DEFAULT_OUTPUT = {
        "width_mm": 200,
        "layer_height_mm": 0.1,
        "layers": 5,
        "base_thickness_mm": 0.4
    }
    
    def __init__(self):
        self._settings_path = self._get_settings_path()
        self._settings = self._load_settings()
    
    def _get_settings_path(self) -> Path:
        """
        Determine settings file path.
        
        For portable builds (GitHub Actions packaged), uses a settings.json
        next to the executable. For normal installation, uses AppData.
        """
        # Check if running as frozen executable (PyInstaller)
        if getattr(os.sys, 'frozen', False):
            # Packaged executable - use portable mode (same directory as exe)
            exe_dir = Path(os.sys.executable).parent
            portable_settings = exe_dir / "settings.json"
            
            # If portable settings exists or directory is writable, use portable mode
            if portable_settings.exists() or os.access(exe_dir, os.W_OK):
                return portable_settings
        
        # Normal mode - use AppData
        appdata = os.environ.get('APPDATA', os.path.expanduser('~'))
        settings_dir = Path(appdata) / "Forge"
        settings_dir.mkdir(parents=True, exist_ok=True)
        return settings_dir / "settings.json"
    
    def _load_settings(self) -> dict:
        """Load settings from JSON file"""
        if self._settings_path.exists():
            try:
                with open(self._settings_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {}
    
    def _save_settings(self):
        """Save settings to JSON file"""
        try:
            with open(self._settings_path, 'w', encoding='utf-8') as f:
                json.dump(self._settings, f, ensure_ascii=False, indent=2)
        except IOError:
            pass  # Silently fail if cannot write
    
    def get_materials(self) -> list[dict]:
        """Get materials configuration"""
        return self._settings.get('materials', self.DEFAULT_MATERIALS.copy())
    
    def set_materials(self, materials: list[dict]):
        """Save materials configuration"""
        self._settings['materials'] = materials
        self._save_settings()
    
    def get_output_settings(self) -> dict:
        """Get output settings"""
        saved = self._settings.get('output', {})
        # Merge with defaults
        result = self.DEFAULT_OUTPUT.copy()
        result.update(saved)
        return result
    
    def set_output_settings(self, output: dict):
        """Save output settings"""
        self._settings['output'] = output
        self._save_settings()


# Global singleton instance
_settings_manager = None

def get_settings_manager() -> SettingsManager:
    """Get the global settings manager instance"""
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager

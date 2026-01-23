import sys
import os
from pathlib import Path
import numpy as np
import cv2

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from forge.core.color_model import ColorModel
from forge.core.exporter import Exporter

def generate_calibration(output_dir: str = "output"):
    # Define CMYW Materials
    # Note: Opacity settings are experimental. Adjust as needed.
    materials = [
        {'name': 'White',   'color': '#FFFFFF', 'opacity': 0.1}, 
        {'name': 'Cyan',    'color': '#00FFFF', 'opacity': 0.6},
        {'name': 'Magenta', 'color': '#FF00FF', 'opacity': 0.6},
        {'name': 'Yellow',  'color': '#FFFF00', 'opacity': 0.6}
    ]
    
    print("Generating Palette for CMYW...")
    # 5 Layers
    model = ColorModel(materials, total_layers=5)
    palette, combinations = model.generate_palette()
    
    total_combos = len(combinations)
    print(f"Total Combinations: {total_combos}")
    
    if total_combos != 1024:
        print("Warning: Expected 1024 combinations for 4 materials / 5 layers.")
    
    # 1. Create Preview Image (32x32)
    grid_size = int(np.sqrt(total_combos)) # Should be 32
    
    # Reshape palette to grid
    img_data = palette.reshape(grid_size, grid_size, 3)
    
    # Create a larger preview for visualization (10x scale)
    scale = 10
    preview = cv2.resize(img_data, (grid_size * scale, grid_size * scale), interpolation=cv2.INTER_NEAREST)
    
    os.makedirs(output_dir, exist_ok=True)
    preview_path = f"{output_dir}/calibration_cmyw_preview.png"
    cv2.imwrite(preview_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    print(f"Preview saved to {preview_path}")
    
    # 2. Export 3MF
    # We need layer_data of shape (H, W, Layers)
    # Convert combinations list to array
    combo_arr = np.array(combinations, dtype=np.uint8) # (1024, 5)
    layer_data = combo_arr.reshape(grid_size, grid_size, 5)
    
    print("Exporting 3MF...")
    try:
        exporter = Exporter()
        # pixel_size_mm = 2.0 -> Each color block is 2x2mm
        target_path = f"{output_dir}/calibration_cmyw.3mf"
        exporter.export(target_path, layer_data, materials, pixel_size_mm=2.0)
        print(f"3MF saved to {target_path}")
    except Exception as e:
        print(f"Failed to export 3MF: {e}")
        print("Ensure Lib3MF is installed via 'uv add lib3mf' or similar.")

if __name__ == "__main__":
    generate_calibration()

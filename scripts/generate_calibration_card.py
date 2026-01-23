import sys
import os
from pathlib import Path
import numpy as np
import cv2
import ctypes

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from forge.core.color_model import ColorModel
from forge.core.exporter import Exporter
from forge.core.optics import calculate_transmitted_color

def generate_calibration_card(output_dir: str = "output"):
    # Define Materials
    materials = [
        {'name': 'White',   'color': '#FFFFFF', 'opacity': 0.1}, 
        {'name': 'Cyan',    'color': '#00FFFF', 'opacity': 0.6},
        {'name': 'Magenta', 'color': '#FF00FF', 'opacity': 0.6},
        {'name': 'Yellow',  'color': '#FFFF00', 'opacity': 0.6}
    ]
    
    # Material Indices mapping
    # 0: White (Base) - implicitly used? 
    # Current Exporter expects material indices 0-3.
    # In our combination definitions below, we will use indices 1, 2, 3 for C, M, Y.
    # Index 0 (White) is usually not "printed" as a layer in the same way, or maybe it is?
    # Let's assume the printer has 4 slots.
    
    # Define the 16 Patches (4x4 Grid)
    # Format: [C_layers, M_layers, Y_layers]
    # We assume max 5 layers total thickness.
    # If sum > 5, it might clip, but our exporter handles fixed layers.
    # Actually, we need to map these counts to a 5-layer stack of material indices.
    
    patches_def = [
        # Row 1: Cyan Gradient
        [1, 0, 0], [3, 0, 0], [5, 0, 0], [0, 0, 0], # Last one empty/white
        
        # Row 2: Magenta Gradient
        [0, 1, 0], [0, 3, 0], [0, 5, 0], [2, 2, 2], # Last one Mix Gray
        
        # Row 3: Yellow Gradient
        [0, 0, 1], [0, 0, 3], [0, 0, 5], [3, 3, 3], # Last one Mix Darker
        
        # Row 4: Secondary & Black
        [0, 5, 5], # Red (M+Y)
        [5, 0, 5], # Green (C+Y)
        [5, 5, 0], # Blue (C+M)
        [5, 5, 5]  # Black (Max)
    ]
    
    # Convert definitions to Layer Stacks (Material Indices)
    # We need to fill a 5-layer array for each patch.
    # Strategy: Fill slots with C, then M, then Y. Remaining slots are White (0) ?
    # Or just empty? If "White" is a material, we fill with 0.
    
    grid_h, grid_w = 4, 4
    total_layers = 5
    
    # Prepare data structures
    layer_data = np.zeros((grid_h, grid_w, total_layers), dtype=np.uint8)
    preview_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
    
    print("Generating 16-color Calibration Card...")
    
    for idx, counts in enumerate(patches_def):
        row = idx // grid_w
        col = idx % grid_w
        
        c_count, m_count, y_count = counts
        
        # Construct primitive layer stack
        # 1 = Cyan, 2 = Magenta, 3 = Yellow, 0 = White
        stack = []
        for _ in range(c_count): stack.append(1)
        for _ in range(m_count): stack.append(2)
        for _ in range(y_count): stack.append(3)
        
        # Fill rest with White (0) up to 5 layers
        while len(stack) < total_layers:
            stack.append(0)
            
        # If > 5 layers (e.g. Black 5,5,5), truncate or handle?
        # Current logic processes top 5.
        # Let's truncate to 5 for the physical print file if necessary, 
        # BUT for color mixing, C5+M5+Y5 implies 15 layers which is impossible in 5 layer height.
        # Wait, lithophane usually has fixed total height?
        # If we print 5 layers total, we can't have 5 C + 5 M.
        # We must limit to 5.
        
        # CORRECT LOGIC:
        # A pixel has a fixed height (e.g. 5 layers).
        # We choose which material goes into which layer.
        # [C, C, M, M, Y] is a valid 5-layer stack.
        # [C, C, C, C, C] is valid.
        
        # So "5 layers of Cyan" means 5 slots are Cyan.
        # "C5 + M5" is impossible in a 5-layer print.
        # We need to adjust our calibration targets to fit in 5 layers!
        
        # Modified Plan for limited layers (Total 5):
        # Gradients: 1, 3, 5 ok.
        # Mixtures: 1C+1M+1Y (3 layers) ok.
        # 2C+2M+1Y (5 layers) ok.
        # Secondary: 2C+3M (5 layers).
        
        real_stack = stack[:total_layers]
        
        # Fill array
        for l in range(total_layers):
            layer_data[row, col, l] = real_stack[l]
            
        # Calculate Preview Color
        # Build layer dicts for optics
        layers_optics = []
        for mat_idx in real_stack:
            m = materials[mat_idx]
            layers_optics.append({
                'color': m['color'],
                'opacity': m['opacity'],
                'thickness': 0.08
            })
            
        rgb = calculate_transmitted_color(layers_optics)
        preview_image[row, col] = rgb
            
    # Save Preview
    os.makedirs(output_dir, exist_ok=True)
    scale = 50 # Make it big
    preview_big = cv2.resize(preview_image, (grid_w * scale, grid_h * scale), interpolation=cv2.INTER_NEAREST)
    
    # Draw simple text labels (optional, skip for now)
    
    preview_path = f"{output_dir}/calibration_card_16.png"
    cv2.imwrite(preview_path, cv2.cvtColor(preview_big, cv2.COLOR_RGB2BGR))
    print(f"Preview saved to {preview_path}")

    # Export 3MF
    print("Exporting 3MF...")
    try:
        exporter = Exporter()
        # pixel_size_mm = 10.0 -> Each block is 10x10mm (Size of a small test patch)
        # Total size = 40x40mm
        target_path = f"{output_dir}/calibration_card_16.3mf"
        exporter.export(target_path, layer_data, materials, pixel_size_mm=10.0)
        print(f"3MF saved to {target_path}")
    except Exception as e:
        print(f"Failed to export 3MF: {e}")

if __name__ == "__main__":
    generate_calibration_card()

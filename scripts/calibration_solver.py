import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

from forge.core.optics import calculate_transmitted_color

def solve_calibration():
    """
    Solves for the best Materials parameters based on observed measurements.
    """
    
    # 1. 定义我们想要校准的材料 (Initial Guess)
    # 这些是当前的默认值，我们将优化它们
    materials = [
        {'name': 'Cyan',    'color_hex': '#00FFFF', 'opacity': 0.6},
        {'name': 'Magenta', 'color_hex': '#FF00FF', 'opacity': 0.6},
        {'name': 'Yellow',  'color_hex': '#FFFF00', 'opacity': 0.6}
        # White 通常作为背景或稀释剂，假设它是常数或稍后加入
    ]
    
    # 2. 录入你的“观测数据” (Observed Data)
    # 格式: (层组合, 实测RGB颜色)
    # 层组合: [C层数, M层数, Y层数]
    # 实测RGB: 你从通过背光照片中吸取的颜色值 (0-255)
    # 
    # === 示例数据 (请替换为你实际测量的结果) ===
    observations = [
        # (层组合[C,M,Y], 实测RGB)
        ([1, 0, 0], (180, 255, 255)), # 1层 Cyan 看起来的样子
        ([5, 0, 0], (0,   200, 200)), # 5层 Cyan 看起来的样子
        ([0, 1, 0], (255, 180, 255)), # 1层 Magenta
        ([0, 5, 0], (200, 0,   200)), # 5层 Magenta
        ([0, 0, 1], (255, 255, 180)), # 1层 Yellow
        ([0, 0, 5], (200, 200, 0)),   # 5层 Yellow
        ([2, 2, 2], (50,  50,  50)),  # 混合灰 (2C+2M+2Y)
    ]
    
    print(f"Calibration with {len(observations)} observations...")

    # Helper: Convert RGB int tuple to 0-1 float array
    def normalize_rgb(rgb_tuple):
        return np.array(rgb_tuple) / 255.0

    # Helper: Parse hex to RGB tuple
    def hex_to_rgb(hex_str):
        h = hex_str.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    
    # Helper: RGB tuple to Hex
    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    # Flatten parameters for optimization
    # Only optimizing Opacity (3 vars) and Color Strength (scaling factor of RGB)?
    # Or optimizing full RGB (3*3 = 9 vars) + Opacity (3 vars) = 12 vars?
    # Simplified approach: Optimize Opacity, and optimize Color "Depth" (Saturating the hue)
    # For advanced users: Optimize full RGB.
    
    # Let's optimize:
    # 1. Opacity (0.0 - 1.0)
    # 2. Color (R, G, B) (0-255)
    # Total 4 params per material * 3 materials = 12 params.
    
    x0 = []
    for mat in materials:
        r, g, b = hex_to_rgb(mat['color_hex'])
        x0.extend([r, g, b, mat['opacity']])
        
    x0 = np.array(x0)
    
    # Loss Function
    def loss_function(params):
        total_error = 0
        
        # Unpack params
        current_mats = []
        for i in range(3): # 3 materials
            base = i * 4
            r, g, b, opacity = params[base:base+4]
            # Constraints soft checking
            r = np.clip(r, 0, 255)
            g = np.clip(g, 0, 255)
            b = np.clip(b, 0, 255)
            opacity = np.clip(opacity, 0.01, 1.0)
            
            current_mats.append({
                'color': (r, g, b),
                'opacity': opacity,
                'thickness': 0.08
            })
            
        # Simulate each observation
        for counts, observed_rgb_tuple in observations:
            measured = normalize_rgb(observed_rgb_tuple)
            
            # Construct layers for simulation
            layers_data = []
            # Cyan
            for _ in range(counts[0]):
                layers_data.append(current_mats[0])
            # Magenta
            for _ in range(counts[1]):
                layers_data.append(current_mats[1])
            # Yellow
            for _ in range(counts[2]):
                layers_data.append(current_mats[2])
                
            # Simulate
            # Assuming white backlight
            simulated_rgb_int = calculate_transmitted_color(layers_data, light_source=(255, 255, 255))
            simulated = simulated_rgb_int / 255.0
            
            # Squared Error
            # Weighted: Human eye is more sensitive to Green? standard Euclidean is okay for now.
            # Using LAB distance would be better but slower.
            error = np.sum((simulated - measured) ** 2)
            total_error += error
            
        return total_error

    # Run Optimization
    # Bounds: RGB (0, 255), Opacity (0, 1)
    bounds = []
    for _ in range(3):
        bounds.extend([(0, 255), (0, 255), (0, 255), (0.01, 1.0)])
        
    print("Optimizing parameters...")
    result = minimize(loss_function, x0, bounds=bounds, method='L-BFGS-B')
    
    print("\n=== Optimized Results ===")
    print(f"Success: {result.success}")
    print(f"Final Loss: {result.fun:.6f}")
    
    optimized_params = result.x
    mat_names = ['Cyan', 'Magenta', 'Yellow']
    
    print("-" * 40)
    for i in range(3):
        base = i * 4
        r, g, b, opacity = optimized_params[base:base+4]
        hex_val = rgb_to_hex((r, g, b))
        print(f"{mat_names[i]}:")
        print(f"  Color: {hex_val} (R={int(r)}, G={int(g)}, B={int(b)})")
        print(f"  Opacity: {opacity:.4f}")
    print("-" * 40)
    print("Copy these values into your analyzer configuration.")

if __name__ == "__main__":
    solve_calibration()

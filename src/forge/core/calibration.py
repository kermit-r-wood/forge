"""
Calibration Core Module
Contains logic for generating calibration cards and solving material parameters.
"""
import numpy as np
import cv2
from scipy.optimize import minimize
from .color_model import ColorModel
from .exporter import Exporter
from .optics import calculate_transmitted_color

class CalibrationGenerator:
    """Generates calibration 3MF and preview images."""
    
    @staticmethod
    def get_16_color_patches():
        """Returns the definitions [C, M, Y] for the 16-patch card."""
        return [
            # Row 1: Cyan Gradient
            [1, 0, 0], [3, 0, 0], [5, 0, 0], [0, 0, 0],
            # Row 2: Magenta Gradient
            [0, 1, 0], [0, 3, 0], [0, 5, 0], [2, 2, 2],
            # Row 3: Yellow Gradient
            [0, 0, 1], [0, 0, 3], [0, 0, 5], [3, 3, 3],
            # Row 4: Secondary & Black
            [0, 5, 5], [5, 0, 5], [5, 5, 0], [5, 5, 5]
        ]
        
    @staticmethod
    def generate_preview(materials: list[dict], patch_defs: list) -> np.ndarray:
        """Generates a generic preview image for the patches."""
        grid_h, grid_w = 4, 4
        preview_image = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for idx, counts in enumerate(patch_defs):
            if idx >= 16: break
            row = idx // grid_w
            col = idx % grid_w
            
            c, m, y = counts
            # Build optics layers
            layers_optics = []
            # Cyan
            for _ in range(c):
                layers_optics.append({'color': materials[1]['color'], 'opacity': materials[1]['opacity'], 'thickness': 0.08})
            # Magenta
            for _ in range(m):
                layers_optics.append({'color': materials[2]['color'], 'opacity': materials[2]['opacity'], 'thickness': 0.08})
            # Yellow
            for _ in range(y):
                layers_optics.append({'color': materials[3]['color'], 'opacity': materials[3]['opacity'], 'thickness': 0.08})
                
            # White filler? Optics function assumes white light source, so empty layers don't change anything unless they have opacity.
            # But the White material (materials[0]) has opacity!
            # We should fill the rest with White layers to match physical print.
            stack_len = c + m + y
            rem = 5 - stack_len
            if rem > 0:
                 for _ in range(rem):
                    layers_optics.append({'color': materials[0]['color'], 'opacity': materials[0]['opacity'], 'thickness': 0.08})
            
            rgb = calculate_transmitted_color(layers_optics)
            preview_image[row, col] = rgb
            
        return preview_image

    @staticmethod
    def export_3mf(file_path: str, materials: list[dict], patch_defs: list):
        """Exports the calibration card to 3MF."""
        grid_h, grid_w = 4, 4
        total_layers = 5
        layer_data = np.zeros((grid_h, grid_w, total_layers), dtype=np.uint8)
        
        for idx, counts in enumerate(patch_defs):
            if idx >= 16: break
            row = idx // grid_w
            col = idx % grid_w
            
            c, m, y = counts
            stack = []
            # Indices: 1=C, 2=M, 3=Y (assuming materials input is [White, C, M, Y])
            for _ in range(c): stack.append(1)
            for _ in range(m): stack.append(2)
            for _ in range(y): stack.append(3)
            
            # Fill rest with White (0)
            while len(stack) < total_layers:
                stack.append(0)
                
            real_stack = stack[:total_layers]
            for l in range(total_layers):
                layer_data[row, col, l] = real_stack[l]
                
        exporter = Exporter()
        exporter.export(file_path, layer_data, materials, pixel_size_mm=10.0)


class CalibrationSolver:
    """Solves for material parameters."""
    
    @staticmethod
    def solve(current_materials: list[dict], observations: list[tuple]) -> list[dict]:
        """
        observations: list of (patch_index, (r, g, b))
        patch_index corresponds to the flat index in get_16_color_patches()
        """
        patch_defs = CalibrationGenerator.get_16_color_patches()
        
        # Determine valid observations
        valid_obs = []
        for p_idx, measured_rgb in observations:
            if 0 <= p_idx < len(patch_defs):
                valid_obs.append((patch_defs[p_idx], measured_rgb))
                
        if not valid_obs:
            raise ValueError("No valid observations provided")
            
        # Optimization target: Opacity and Color Strength (RGB scaling) for C, M, Y
        # We assume White (index 0) is constant for simplicity, or we optimize it too?
        # Let's optimize C(1), M(2), Y(3).
        
        def hex_to_rgb(hex_str):
            h = hex_str.lstrip('#')
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        
        # Initial guess (x0)
        # 4 params per material (R, G, B, Opacity) * 3 materials = 12 params
        x0 = []
        for i in range(1, 4): # Indices 1, 2, 3
            mat = current_materials[i]
            r, g, b = hex_to_rgb(mat['color'])
            x0.extend([r, g, b, mat['opacity']])
            
        def loss_function(params):
            total_error = 0
            
            temp_materials = [current_materials[0].copy()] # copy White
            
            # Reconstruct C, M, Y materials
            for i in range(3):
                base = i * 4
                r, g, b, opacity = params[base:base+4]
                r = np.clip(r, 0, 255)
                g = np.clip(g, 0, 255)
                b = np.clip(b, 0, 255)
                opacity = np.clip(opacity, 0.01, 1.0)
                
                temp_materials.append({
                    'color': (r, g, b),
                    'opacity': opacity,
                    'thickness': 0.08
                })
                
            for counts, measured_rgb in valid_obs:
                c, m, y = counts
                layers_optics = []
                # Rebuild stack logic
                # Cyan (Index 1 in temp_materials)
                for _ in range(c):
                     layers_optics.append(temp_materials[1])
                for _ in range(m):
                     layers_optics.append(temp_materials[2])
                for _ in range(y):
                     layers_optics.append(temp_materials[3])
                
                rem = 5 - (c+m+y)
                if rem > 0:
                    for _ in range(rem):
                        layers_optics.append({'color': hex_to_rgb(temp_materials[0]['color']), 'opacity': temp_materials[0]['opacity'], 'thickness': 0.08})
                        
                simulated_rgb = calculate_transmitted_color(layers_optics)
                
                # Error (MSE)
                sim_norm = simulated_rgb / 255.0
                meas_norm = np.array(measured_rgb) / 255.0
                total_error += np.sum((sim_norm - meas_norm) ** 2)
                
            return total_error

        # Bounds
        bounds = []
        for _ in range(3):
            bounds.extend([(0, 255), (0, 255), (0, 255), (0.01, 1.0)])
            
        result = minimize(loss_function, x0, bounds=bounds, method='L-BFGS-B')
        
        # Apply results
        optimized = []
        optimized.append(current_materials[0]) # Keep White
        
        final_params = result.x
        for i in range(3):
            base = i * 4
            r, g, b, opacity = final_params[base:base+4]
            hex_col = '#{:02x}{:02x}{:02x}'.format(int(r), int(g), int(b))
            
            original_name = current_materials[i+1]['name']
            optimized.append({
                'name': original_name,
                'color': hex_col,
                'opacity': float(opacity)
            })
            
        return optimized

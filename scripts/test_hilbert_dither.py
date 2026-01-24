
import numpy as np
import cv2
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

try:
    from forge.core.dithering.riemersma import RiemersmaDither
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_hilbert():
    print("Creating dummy image...")
    # Create a gradient image 200x200
    w, h = 200, 200
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = [x % 255, y % 255, (x+y) % 255]
            
    # Dummy palette (1024 colors)
    print("Generating large palette (1024 colors)...")
    colors = []
    for r in range(0, 256, 32):
        for g in range(0, 256, 32):
            for b in range(0, 256, 64):
                colors.append([r, g, b])
    palette = np.array(colors[:1024], dtype=np.uint8)
    print(f"Palette size: {len(palette)}")
    
    print("Initializing dither...")
    dither = RiemersmaDither()
    
    print("Applying dither...")
    try:
        result = dither.apply(img, palette)
        print("Dither applied successfully")
        print(f"Result shape: {result.shape}")
        
        # Verify result content
        unique_colors = np.unique(result.reshape(-1, 3), axis=0)
        print(f"Unique colors count: {len(unique_colors)}")
        print(f"Unique colors:\n{unique_colors}")
        
    except Exception as e:
        print(f"Error applying dither: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_hilbert()

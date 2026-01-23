import pytest
import numpy as np
import cv2
from forge.core.color_distance import cie76_distance, cie94_distance, ciede2000_distance
from forge.core.analyzer import Analyzer
from forge.core.dithering.base import BaseDither

def test_distance_metrics():
    # Test colors (LAB)
    # Red-ish
    c1 = np.array([50.0, 80.0, 60.0])
    # Slightly different Red
    c2 = np.array([50.0, 82.0, 62.0])
    
    d76 = cie76_distance(c1, c2)
    d94 = cie94_distance(c1, c2)
    d00 = ciede2000_distance(c1, c2)
    
    print(f"Distances: CIE76={d76}, CIE94={d94}, CIEDE2000={d00}")
    
    assert d76 > 0
    assert d94 > 0
    assert d00 > 0
    assert d76 != d00 # Should be different

def test_analyzer_integration(tmp_path):
    # Setup
    analyzer = Analyzer()
    img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    # Save temp image because load_image expects path
    img_path = tmp_path / "test_opt.png"
    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    analyzer.load_image(img_path)
    
    materials = [
        {"name": "White", "color": "#FFFFFF", "opacity": 0.3},
        {"name": "Red", "color": "#FF0000", "opacity": 0.7},
        {"name": "Yellow", "color": "#FFFF00", "opacity": 0.6},
        {"name": "Blue", "color": "#0000FF", "opacity": 0.7},
    ]
    
    # Test CIE76
    settings_76 = {
        "dither": 0, # Floyd-Steinberg
        "distance_metric": "cie76"
    }
    analyzer.process(settings_76, materials, width_mm=20)
    assert analyzer.indices is not None
    
    # Test CIEDE2000
    settings_00 = {
        "dither": 0,
        "distance_metric": "ciede2000"
    }
    analyzer.process(settings_00, materials, width_mm=20)
    assert analyzer.indices is not None
    
    # Test No Dither + CIEDE2000 (Batch mode)
    settings_nodither = {
        "dither": 3, # None
        "distance_metric": "ciede2000"
    }
    analyzer.process(settings_nodither, materials, width_mm=20)
    assert analyzer.indices is not None

def test_hybrid_search_logic():
    # Mock BaseDither to test finding logic specifically
    class MockDither(BaseDither):
        def apply(self, image, palette):
            pass
            
    dither = MockDither(distance_metric='ciede2000')
    palette = np.array([
        [0, 0, 0],       # Black
        [255, 255, 255], # White
        [255, 0, 0]      # Red
    ], dtype=np.uint8)
    
    # Pixel close to Red
    pixel = np.array([250, 10, 10], dtype=np.uint8)
    
    closest = dither.find_closest_color(pixel, palette)
    assert np.array_equal(closest, [255, 0, 0])

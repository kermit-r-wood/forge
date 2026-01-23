"""
集成测试
"""
import pytest
import numpy as np
from pathlib import Path
from forge.core.analyzer import Analyzer
from forge.core.color_model import ColorModel
from forge.core.exporter import Exporter

@pytest.fixture
def sample_image():
    # 创建一个简单的 100x100 RGB 图像
    return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def materials():
    return [
        {"name": "White", "color": "#FFFFFF", "opacity": 0.3},
        {"name": "Red", "color": "#FF0000", "opacity": 0.7},
        {"name": "Yellow", "color": "#FFFF00", "opacity": 0.6},
        {"name": "Blue", "color": "#0000FF", "opacity": 0.7},
    ]

def test_color_model(materials):
    model = ColorModel(materials, layer_height=0.08, total_layers=5)
    palette, combinations = model.generate_palette()
    assert len(palette) == 4 ** 5 # 1024
    assert palette.shape == (1024, 3)
    assert len(combinations) == 1024

def test_analyzer_flow(sample_image, materials, tmp_path):
    # 保存 sample image
    img_path = tmp_path / "test.png"
    import cv2
    cv2.imwrite(str(img_path), sample_image)
    
    analyzer = Analyzer()
    analyzer.load_image(img_path)
    assert analyzer.image is not None
    
    settings = {
        "preprocess": 0, # Bilateral
        "quantize": 0,   # KMeans
        "dither": 0      # Floyd-Steinberg
    }
    
    analyzer.process(settings, materials, width_mm=100)
    
    assert analyzer.processed is not None
    assert analyzer.indices is not None
    assert analyzer.indices.shape == (250, 250) # 100mm / 0.4 = 250
    
    layer_data = analyzer.get_layer_data()
    assert layer_data.shape == (250, 250, 5)

def test_exporter(tmp_path):
    # Mock layer data
    layer_data = np.zeros((10, 10, 5), dtype=np.uint8)
    materials = [{"name": "Test", "color": (255,255,255), "opacity": 1.0}]
    
    exporter = Exporter()
    out_path = tmp_path / "output.3mf"
    
    try:
        exporter.export(str(out_path), layer_data, materials)
        assert out_path.exists()
        assert out_path.stat().st_size > 0
    except OSError as e:
        # Lib3MF not found or load error
        pytest.skip(f"Lib3MF error: {e}")
    except RuntimeError as e:
        if "library not found" in str(e).lower():
             pytest.skip("Lib3MF native library issue")
        else:
            raise e

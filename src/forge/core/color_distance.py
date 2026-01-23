"""
Color Distance Algorithms

Provides various algorithms to calculate the perceptual difference between two colors
in the CIELAB color space.

Algorithms:
- CIE76 (Euclidean): Fast, but perceptually non-uniform.
- CIE94: Better perceptual uniformity, especially for saturated colors.
- CIEDE2000: The current standard for perceptual accuracy, computationally expensive.
- Weighted Euclidean: Allows custom weighting of L, a, b components.
"""
import numpy as np

def cie76_distance(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Calculate CIE76 color difference (Euclidean distance in LAB).
    
    :param lab1: Array of LAB colors (N, 3) or (3,)
    :param lab2: Array of LAB colors (N, 3) or (3,)
    :return: Array of distances
    """
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff**2, axis=-1))

def weighted_lab_distance(lab1: np.ndarray, lab2: np.ndarray, weights=(1.0, 1.0, 1.0)) -> np.ndarray:
    """
    Calculate Weighted Euclidean distance in LAB.
    Useful to prioritize Lightness over Chroma.
    
    :param weights: Tuple of (wL, wa, wb)
    """
    diff = lab1 - lab2
    w = np.array(weights)
    return np.sqrt(np.sum((diff * w)**2, axis=-1))

def cie94_distance(lab1: np.ndarray, lab2: np.ndarray, kL=1, kC=1, kH=1, K1=0.045, K2=0.015) -> np.ndarray:
    """
    Calculate CIE94 color difference.
    
    :param lab1: Reference color(s) (N, 3)
    :param lab2: Sample color(s) (N, 3)
    """
    # Handle single color inputs
    if lab1.ndim == 1: lab1 = lab1.reshape(1, 3)
    if lab2.ndim == 1: lab2 = lab2.reshape(1, 3)
    
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    
    dL = L1 - L2
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    dC = C1 - C2
    
    # da = a1 - a2
    # db = b1 - b2
    # dH2 = da**2 + db**2 - dC**2 
    # Use simpler calculation for dH
    # Delta H = sqrt(Delta E^2 - Delta L^2 - Delta C^2)
    # But strictly dH^2 = dE_76^2 - dL^2 - dC^2
    dE_76 = np.sum((lab1 - lab2)**2, axis=-1)
    dH2 = dE_76 - dL**2 - dC**2
    # Avoid negative values due to float precision
    dH2 = np.maximum(dH2, 0)
    
    SL = 1
    SC = 1 + K1 * C1
    SH = 1 + K2 * C1
    
    final_L = dL / (kL * SL)
    final_C = dC / (kC * SC)
    final_H2 = dH2 / ((kH * SH)**2)
    
    return np.sqrt(final_L**2 + final_C**2 + final_H2)

def ciede2000_distance(lab1: np.ndarray, lab2: np.ndarray, kL=1, kC=1, kH=1) -> np.ndarray:
    """
    Calculate CIEDE2000 color difference.
    Vectorized implementation.
    
    Reference: http://www2.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
    
    :param kL: Weight for Lightness (usually 1, use 2 for textiles/texture tolerance)
    :param kC: Weight for Chroma
    :param kH: Weight for Hue
    """
    # Ensure inputs are 2D arrays (N, 3)
    # If comparing 1 pixel to N palette colors, broadcast appropriately
    if lab1.ndim == 1: lab1 = lab1[np.newaxis, :]
    if lab2.ndim == 1: lab2 = lab2[np.newaxis, :]
    
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]
    
    # kL, kC, kH passed as args
    
    # 1. Calculate C_prime, a_prime
    avg_L = (L1 + L2) / 2.0
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    
    G = 0.5 * (1 - np.sqrt(avg_C**7 / (avg_C**7 + 25**7)))
    
    a1_p = (1 + G) * a1
    a2_p = (1 + G) * a2
    
    C1_p = np.sqrt(a1_p**2 + b1**2)
    C2_p = np.sqrt(a2_p**2 + b2**2)
    
    # 2. Calculate dL_p, dC_p, dH_p
    h1_p = np.arctan2(b1, a1_p)
    h1_p = np.degrees(h1_p) % 360
    
    h2_p = np.arctan2(b2, a2_p)
    h2_p = np.degrees(h2_p) % 360
    
    dL_p = L2 - L1
    dC_p = C2_p - C1_p
    
    dh_p = h2_p - h1_p
    dh_p = np.where(np.abs(dh_p) > 180, dh_p - 360 * np.sign(dh_p), dh_p)
    dH_p = 2 * np.sqrt(C1_p * C2_p) * np.sin(np.radians(dh_p) / 2.0)
    
    # 3. Calculate RT, SL, SC, SH
    avg_L_p = (L1 + L2) / 2.0
    avg_C_p = (C1_p + C2_p) / 2.0
    
    # avg_h_p calculation with condition
    sum_h_p = h1_p + h2_p
    # If absolute difference > 180, modify sum
    abs_diff_h = np.abs(h1_p - h2_p)
    avg_h_p = np.where(abs_diff_h > 180, 
                       np.where(sum_h_p < 360, sum_h_p + 360, sum_h_p - 360) / 2.0,
                       sum_h_p / 2.0)
    
    T = 1 - 0.17 * np.cos(np.radians(avg_h_p - 30)) + \
        0.24 * np.cos(np.radians(2 * avg_h_p)) + \
        0.32 * np.cos(np.radians(3 * avg_h_p + 6)) - \
        0.20 * np.cos(np.radians(4 * avg_h_p - 63))
        
    delta_theta = 30 * np.exp(-((avg_h_p - 275) / 25)**2)
    RC = 2 * np.sqrt(avg_C_p**7 / (avg_C_p**7 + 25**7))
    RT = -np.sin(np.radians(2 * delta_theta)) * RC
    
    SL = 1 + (0.015 * (avg_L_p - 50)**2) / np.sqrt(20 + (avg_L_p - 50)**2)
    SC = 1 + 0.045 * avg_C_p
    SH = 1 + 0.015 * avg_C_p * T
    
    # 4. Calculate Delta E
    term1 = (dL_p / (kL * SL))**2
    term2 = (dC_p / (kC * SC))**2
    term3 = (dH_p / (kH * SH))**2
    term4 = RT * (dC_p / (kC * SC)) * (dH_p / (kH * SH))
    
    return np.sqrt(term1 + term2 + term3 + term4)


# ============================================================
# OKLab Color Space
# A modern perceptually uniform color space, more accurate than LAB
# Reference: https://bottosson.github.io/posts/oklab/
# ============================================================

def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """
    Convert sRGB to OKLab color space.
    
    :param rgb: RGB colors (N, 3) or (H, W, 3) in range [0, 255]
    :return: OKLab colors with same shape, L in [0, 1], a/b in ~[-0.4, 0.4]
    """
    original_shape = rgb.shape
    rgb_flat = rgb.reshape(-1, 3).astype(np.float64) / 255.0
    
    # sRGB to linear RGB
    linear = np.where(rgb_flat <= 0.04045, 
                      rgb_flat / 12.92, 
                      ((rgb_flat + 0.055) / 1.055) ** 2.4)
    
    # Linear RGB to LMS (using Oklab matrix)
    M1 = np.array([
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005]
    ])
    
    lms = linear @ M1.T
    
    # Cube root
    lms_cbrt = np.sign(lms) * np.abs(lms) ** (1/3)
    
    # LMS to OKLab
    M2 = np.array([
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660]
    ])
    
    oklab = lms_cbrt @ M2.T
    
    return oklab.reshape(original_shape).astype(np.float32)


def oklab_to_rgb(oklab: np.ndarray) -> np.ndarray:
    """
    Convert OKLab to sRGB color space.
    
    :param oklab: OKLab colors (N, 3) or (H, W, 3)
    :return: RGB colors with same shape, in range [0, 255]
    """
    original_shape = oklab.shape
    oklab_flat = oklab.reshape(-1, 3).astype(np.float64)
    
    # OKLab to LMS
    M2_inv = np.array([
        [1.0, 0.3963377774, 0.2158037573],
        [1.0, -0.1055613458, -0.0638541728],
        [1.0, -0.0894841775, -1.2914855480]
    ])
    
    lms_cbrt = oklab_flat @ M2_inv.T
    
    # Cube
    lms = lms_cbrt ** 3
    
    # LMS to linear RGB
    M1_inv = np.array([
        [4.0767416621, -3.3077115913, 0.2309699292],
        [-1.2684380046, 2.6097574011, -0.3413193965],
        [-0.0041960863, -0.7034186147, 1.7076147010]
    ])
    
    linear = lms @ M1_inv.T
    
    # Linear RGB to sRGB
    srgb = np.where(linear <= 0.0031308,
                    linear * 12.92,
                    1.055 * (np.maximum(linear, 0) ** (1/2.4)) - 0.055)
    
    # Clip and convert to uint8
    srgb = np.clip(srgb * 255, 0, 255).astype(np.uint8)
    
    return srgb.reshape(original_shape)


def oklab_distance(oklab1: np.ndarray, oklab2: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance in OKLab color space.
    Since OKLab is perceptually uniform, Euclidean distance is meaningful.
    
    :param oklab1: OKLab colors (N, 3) or (3,)
    :param oklab2: OKLab colors (N, 3) or (3,)
    :return: Array of distances
    """
    diff = oklab1 - oklab2
    return np.sqrt(np.sum(diff**2, axis=-1))


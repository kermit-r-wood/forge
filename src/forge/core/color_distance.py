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
import math
from numba import jit, prange

def cie76_distance(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """
    Calculate CIE76 color difference (Euclidean distance in LAB).
    
    :param lab1: Array of LAB colors (N, 3) or (3,)
    :param lab2: Array of LAB colors (N, 3) or (3,)
    :return: Array of distances
    """
    diff = lab1 - lab2
    return np.sqrt(np.sum(diff**2, axis=-1))




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


@jit(nopython=True, fastmath=True, cache=True)
def _ciede2000_scalar(L1, a1, b1, L2, a2, b2):
    """
    Calculate CIEDE2000 color difference between two scalar LAB colors.
    Used for Numba parallel loops.
    """
    avg_L = (L1 + L2) / 2.0
    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)
    avg_C = (C1 + C2) / 2.0
    
    avg_C7 = avg_C**7
    G = 0.5 * (1.0 - math.sqrt(avg_C7 / (avg_C7 + 6103515625.0))) # 25**7 = 6103515625.0
    
    a1_p = (1.0 + G) * a1
    a2_p = (1.0 + G) * a2
    
    C1_p = math.sqrt(a1_p**2 + b1**2)
    C2_p = math.sqrt(a2_p**2 + b2**2)
    
    h1_p = math.atan2(b1, a1_p)
    h1_p = math.degrees(h1_p) % 360.0
    
    h2_p = math.atan2(b2, a2_p)
    h2_p = math.degrees(h2_p) % 360.0
    
    dL_p = L2 - L1
    dC_p = C2_p - C1_p
    
    dh_p = h2_p - h1_p
    if abs(dh_p) > 180.0:
        if dh_p > 0.0:
            dh_p -= 360.0
        else:
            dh_p += 360.0
            
    dH_p = 2.0 * math.sqrt(C1_p * C2_p) * math.sin(math.radians(dh_p) / 2.0)
    
    avg_L_p = (L1 + L2) / 2.0
    avg_C_p = (C1_p + C2_p) / 2.0
    
    sum_h_p = h1_p + h2_p
    abs_diff_h = abs(h1_p - h2_p)
    if abs_diff_h > 180.0:
        if sum_h_p < 360.0:
            avg_h_p = (sum_h_p + 360.0) / 2.0
        else:
            avg_h_p = (sum_h_p - 360.0) / 2.0
    else:
        avg_h_p = sum_h_p / 2.0
        
    T = 1.0 - 0.17 * math.cos(math.radians(avg_h_p - 30.0)) + \
        0.24 * math.cos(math.radians(2.0 * avg_h_p)) + \
        0.32 * math.cos(math.radians(3.0 * avg_h_p + 6.0)) - \
        0.20 * math.cos(math.radians(4.0 * avg_h_p - 63.0))
        
    delta_theta = 30.0 * math.exp(-((avg_h_p - 275.0) / 25.0)**2)
    avg_C_p7 = avg_C_p**7
    RC = 2.0 * math.sqrt(avg_C_p7 / (avg_C_p7 + 6103515625.0))
    RT = -math.sin(math.radians(2.0 * delta_theta)) * RC
    
    SL = 1.0 + (0.015 * (avg_L_p - 50.0)**2) / math.sqrt(20.0 + (avg_L_p - 50.0)**2)
    SC = 1.0 + 0.045 * avg_C_p
    SH = 1.0 + 0.015 * avg_C_p * T
    
    term1 = (dL_p / SL)**2
    term2 = (dC_p / SC)**2
    term3 = (dH_p / SH)**2
    term4 = RT * (dC_p / SC) * (dH_p / SH)
    
    return math.sqrt(term1 + term2 + term3 + term4)


@jit(nopython=True, fastmath=True, parallel=True, cache=True)
def match_colors_ciede2000_numba(image_lab_flat, palette_lab):
    """
    Fast, parallel matching of colors using Numba-accelerated CIEDE2000 metric.
    """
    n_pixels = image_lab_flat.shape[0]
    n_palette = palette_lab.shape[0]
    indices = np.zeros(n_pixels, dtype=np.int32)
    
    for i in prange(n_pixels):
        best_dist = 1e10
        best_idx = 0
        
        L1 = image_lab_flat[i, 0]
        a1 = image_lab_flat[i, 1]
        b1 = image_lab_flat[i, 2]
        
        for j in range(n_palette):
            L2 = palette_lab[j, 0]
            a2 = palette_lab[j, 1]
            b2 = palette_lab[j, 2]
            
            dist = _ciede2000_scalar(L1, a1, b1, L2, a2, b2)
            if dist < best_dist:
                best_dist = dist
                best_idx = j
                
        indices[i] = best_idx
        
    return indices


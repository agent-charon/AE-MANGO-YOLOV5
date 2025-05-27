import numpy as np
import math

def calculate_ellipse_axes_from_bbox(bbox_w_pixels, bbox_h_pixels):
    """
    Calculates the major (a) and minor (b) axes of an ellipse inscribed
    in a bounding box using Eq. 6 from the paper.
    a = sqrt(w^2 + h^2) / 2
    b = (2 * w * h) / (pi * sqrt(w^2 + h^2))
    
    Args:
        bbox_w_pixels (float or np.ndarray): Width of the bounding box in pixels.
        bbox_h_pixels (float or np.ndarray): Height of the bounding box in pixels.
        
    Returns:
        tuple: (major_axis_a, minor_axis_b) in pixels.
    """
    w_sq = bbox_w_pixels**2
    h_sq = bbox_h_pixels**2
    sqrt_w_sq_plus_h_sq = np.sqrt(w_sq + h_sq)

    major_axis_a = sqrt_w_sq_plus_h_sq / 2.0
    
    # Ensure denominator is not zero for minor axis b
    denominator_b = np.pi * sqrt_w_sq_plus_h_sq
    # Add a small epsilon to prevent division by zero if w and h are both zero
    minor_axis_b = (2 * bbox_w_pixels * bbox_h_pixels) / (denominator_b + 1e-9)
    
    return major_axis_a, minor_axis_b

def calculate_ellipse_area(major_axis_a, minor_axis_b):
    """
    Calculates the area of an ellipse.
    Area = pi * a * b
    
    Args:
        major_axis_a (float or np.ndarray): Major axis.
        minor_axis_b (float or np.ndarray): Minor axis.
        
    Returns:
        float or np.ndarray: Area of the ellipse.
    """
    return np.pi * major_axis_a * minor_axis_b

def calculate_ellipse_eccentricity(major_axis_a, minor_axis_b):
    """
    Calculates the eccentricity 'e' of an ellipse.
    e = sqrt(1 - (b/a)^2)
    
    Args:
        major_axis_a (float or np.ndarray): Major axis.
        minor_axis_b (float or np.ndarray): Minor axis.
        
    Returns:
        float or np.ndarray: Eccentricity of the ellipse. Returns 0 if a is 0.
    """
    # Ensure major_axis_a is not zero and b <= a
    # If a is zero, eccentricity is undefined or 0. If b > a, ratio > 1, sqrt becomes complex.
    # The formulas for a and b from bbox ensure a >= b (approximately, due to pi).
    # Diagonal/2 is 'a'. (2wh)/(pi*diag) is 'b'.
    # (b/a) = (4wh)/(pi*(w^2+h^2)). Max value of (wh)/(w^2+h^2) is 1/2 (when w=h). So b/a <= 2/pi < 1.
    
    ratio_sq = np.zeros_like(major_axis_a, dtype=float)
    # Create a mask for non-zero major_axis_a
    valid_mask = major_axis_a > 1e-9
    
    # Compute (b/a)^2 only for valid entries
    # Ensure b is not greater than a which can happen due to floating point issues if w or h is tiny
    b_over_a = np.zeros_like(major_axis_a, dtype=float)
    b_over_a[valid_mask] = np.minimum(minor_axis_b[valid_mask] / major_axis_a[valid_mask], 1.0) 
    ratio_sq[valid_mask] = b_over_a[valid_mask]**2
            
    eccentricity_sq = 1.0 - ratio_sq
    # Clamp to avoid small negative numbers due to precision before sqrt
    eccentricity = np.sqrt(np.maximum(eccentricity_sq, 0.0))
    return eccentricity

def ellipse_perimeter_ramanujan_approx(a, b):
    """
    Calculates the approximate perimeter of an ellipse using Ramanujan's approximation.
    P approx = pi * [3(a+b) - sqrt((3a+b)(a+3b))]
    
    Args:
        a (float or np.ndarray): Major semi-axis.
        b (float or np.ndarray): Minor semi-axis.
        
    Returns:
        float or np.ndarray: Approximate perimeter.
    """
    term1 = 3 * (a + b)
    term2_factor1 = (3 * a + b)
    term2_factor2 = (a + 3 * b)
    # Ensure factors are non-negative before sqrt
    term2 = np.sqrt(np.maximum(term2_factor1 * term2_factor2, 0.0))
    perimeter = np.pi * (term1 - term2)
    return perimeter


if __name__ == '__main__':
    w_px, h_px = 50.0, 50.0 # Square bounding box
    a, b = calculate_ellipse_axes_from_bbox(w_px, h_px)
    print(f"For bbox {w_px}x{h_px}: a_px={a:.2f}, b_px={b:.2f}")
    area = calculate_ellipse_area(a, b)
    print(f"Ellipse Area (pixels^2): {area:.2f}")
    ecc = calculate_ellipse_eccentricity(a, b)
    print(f"Eccentricity: {ecc:.2f}") # Should be 0 for circle (w=h -> a=diag/2, b=2a/pi)
                                     # My b calc: 2*w*h / (pi * sqrt(w^2+h^2))
                                     # For w=h, b = 2w^2 / (pi * w*sqrt(2)) = sqrt(2)w/pi = (2a)/pi
                                     # So e = sqrt(1 - (2/pi)^2) approx 0.77 for inscribed ellipse in square.
                                     # The paper implies this ellipse *is* the mango shape.
    perimeter = ellipse_perimeter_ramanujan_approx(a,b)
    print(f"Perimeter (Ramanujan approx pixels): {perimeter:.2f}")


    w_px, h_px = 80.0, 40.0 # Rectangular bounding box
    a, b = calculate_ellipse_axes_from_bbox(w_px, h_px)
    print(f"\nFor bbox {w_px}x{h_px}: a_px={a:.2f}, b_px={b:.2f}")
    area = calculate_ellipse_area(a, b)
    print(f"Ellipse Area (pixels^2): {area:.2f}")
    ecc = calculate_ellipse_eccentricity(a, b)
    print(f"Eccentricity: {ecc:.2f}")
    perimeter = ellipse_perimeter_ramanujan_approx(a,b)
    print(f"Perimeter (Ramanujan approx pixels): {perimeter:.2f}")
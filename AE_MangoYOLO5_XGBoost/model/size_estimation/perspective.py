import numpy as np

def calculate_perspective_factor_f(focal_length_pixels, real_height_object_cm, 
                                   object_height_in_image_pixels, distance_to_object_cm):
    """
    Calculates the perspective adjustment factor 'f' using Eq. 7 from the paper.
    f = (Focal Length * Real Height of Object) / (Object Height in Image * Distance to Object)
    
    Args:
        focal_length_pixels (float): Effective focal length of the camera in pixels.
        real_height_object_cm (float): Actual height of the object (or a reference object) in cm.
        object_height_in_image_pixels (float): Height of the object in the image in pixels.
        distance_to_object_cm (float): Distance from camera to the object in cm.
        
    Returns:
        float: Perspective adjustment factor 'f'.
    """
    numerator = focal_length_pixels * real_height_object_cm
    denominator = object_height_in_image_pixels * distance_to_object_cm
    
    if denominator == 0:
        # print("Warning: Denominator for perspective factor 'f' is zero. Returning 1.0.")
        return 1.0 # Avoid division by zero, though this indicates an issue.
    return numerator / denominator

def apply_perspective_correction(axis_a_pixels, axis_b_pixels, perspective_factor_f):
    """
    Applies the perspective adjustment factor 'f' to the ellipse axes.
    a' = f * a
    b' = f * b
    
    Args:
        axis_a_pixels (float or np.ndarray): Major axis 'a' in pixels.
        axis_b_pixels (float or np.ndarray): Minor axis 'b' in pixels.
        perspective_factor_f (float): Perspective adjustment factor.
        
    Returns:
        tuple: (corrected_axis_a, corrected_axis_b) which are now in cm if 'f' is set up correctly.
    """
    # The factor 'f' should have units of cm/pixel if set up as per Eq. 7.
    # So a_prime and b_prime will be in cm.
    corrected_axis_a = perspective_factor_f * axis_a_pixels
    corrected_axis_b = perspective_factor_f * axis_b_pixels
    return corrected_axis_a, corrected_axis_b

def get_pixel_to_cm_ratio(known_object_real_size_cm, known_object_image_size_pixels, distance_to_object_m, camera_focal_length_mm, sensor_width_mm, image_width_pixels):
    """
    Alternative way to get cm/pixel if 'f' is hard to calibrate directly with a mango.
    This can be done with a reference object of known size at the same distance.
    
    pixels_per_cm = (object_pixels * focal_length_pixels) / (object_real_cm * distance_pixels) -> not simple.
    
    Simpler: cm_per_pixel = (object_size_cm * distance_m) / (object_size_pixels * focal_length_m) -- from pinhole
    Or, more directly from a reference object:
    cm_per_pixel_at_distance = RealSize_cm / ImageSize_pixels
    This simple ratio only works if the object is on a plane perpendicular to camera axis and perspective distortion is minimal.
    The paper's 'f' accounts for this.
    
    If a fixed distance (5m) is used, one could calibrate a cm/pixel factor at that distance.
    pixel_size_at_5m_cm = (sensor_element_size_mm * 5000_mm / focal_length_mm) / 10 (to cm)
    """
    # This function is more for conceptual understanding. The paper uses factor 'f'.
    # If we have a reference object:
    if known_object_image_size_pixels > 0:
        cm_per_pixel = known_object_real_size_cm / known_object_image_size_pixels
        return cm_per_pixel
    return None # Cannot determine

if __name__ == '__main__':
    # Example usage of perspective factor 'f'
    # These are hypothetical values
    focal_L_px = 1500.0  # Camera effective focal length in pixels
    real_mango_h_cm = 10.0 # Assume we know this for one mango (for calibration of 'f')
    mango_h_img_px = 100.0 # Its height in image pixels
    dist_cm = 500.0      # 5 meters
    
    f = calculate_perspective_factor_f(focal_L_px, real_mango_h_cm, mango_h_img_px, dist_cm)
    print(f"Calculated perspective factor 'f': {f:.4f} (cm/pixel)")
    
    # Now use this 'f' for other mangoes detected at the same distance
    detected_bbox_w_px = 80.0
    detected_bbox_h_px = 90.0
    
    from model.size_estimation.ellipse_utils import calculate_ellipse_axes_from_bbox
    a_px, b_px = calculate_ellipse_axes_from_bbox(detected_bbox_w_px, detected_bbox_h_px)
    print(f"Detected ellipse axes in pixels: a_px={a_px:.2f}, b_px={b_px:.2f}")
    
    a_prime_cm, b_prime_cm = apply_perspective_correction(a_px, b_px, f)
    print(f"Corrected ellipse axes in cm: a'_cm={a_prime_cm:.2f}, b'_cm={b_prime_cm:.2f}")

    # --- Alternative: Simple cm/pixel calibration at fixed distance ---
    # Suppose you placed a 10cm ruler at 5m and it measured 20 pixels.
    ref_obj_cm = 10.0
    ref_obj_px = 20.0 # This would be a very low resolution or very far object for 10cm
    
    # This simple cm_per_pixel doesn't use focal length or exact perspective geometry of 'f'
    # It's a rough approximation. The paper's 'f' is more robust.
    # simple_cm_per_pixel = ref_obj_cm / ref_obj_px 
    # print(f"\nSimple cm/pixel at 5m (example): {simple_cm_per_pixel:.4f} cm/pixel")
    # a_cm_simple = a_px * simple_cm_per_pixel
    # b_cm_simple = b_px * simple_cm_per_pixel
    # print(f"Simple estimated axes in cm: a_cm={a_cm_simple:.2f}, b_cm={b_cm_simple:.2f}")

    # The paper's approach using 'f' (Eq. 7) is preferred.
    # "To convert this to the actual size, we accounted for parameters such as
    # the consistently maintained camera-to-tree distance of 5 meters."
    # This suggests that the perspective factor 'f' or an equivalent pixel-to-cm conversion
    # at that distance was established.

    # If f is set from regression.yaml as a fixed value:
    fixed_f_from_config = 0.12 # Example: if 1 pixel at 5m = 0.12 cm for this camera setup
    a_prime_cm_fixed_f, b_prime_cm_fixed_f = apply_perspective_correction(a_px, b_px, fixed_f_from_config)
    print(f"\nCorrected ellipse axes in cm (using fixed f={fixed_f_from_config}): a'_cm={a_prime_cm_fixed_f:.2f}, b'_cm={b_prime_cm_fixed_f:.2f}")
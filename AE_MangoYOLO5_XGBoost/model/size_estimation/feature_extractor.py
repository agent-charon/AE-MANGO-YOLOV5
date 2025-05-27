import numpy as np
from .ellipse_utils import (
    calculate_ellipse_axes_from_bbox, 
    calculate_ellipse_eccentricity
)
from .perspective import apply_perspective_correction #, calculate_perspective_factor_f

class SizeFeatureExtractor:
    def __init__(self, perspective_config):
        """
        Args:
            perspective_config (dict): Configuration for perspective correction.
                Expected keys: 'focal_length_pixels', 'real_height_object_cm_for_f_calib', (if f is dynamic)
                               'object_height_in_image_pixels_for_f_calib', 'distance_to_object_m'
                               OR 'perspective_factor_f' if 'f' is pre-calibrated and fixed.
        """
        self.config = perspective_config
        self.perspective_factor_f = perspective_config.get('perspective_factor_f', None)
        if self.perspective_factor_f is None:
            # TODO: Implement dynamic 'f' calculation if needed, or ensure 'f' is always provided.
            # For now, assume 'f' is provided or can be calculated based on knowns.
            # The paper implies a fixed 5m distance, so 'f' might be calibrated once for this setup.
            print("Warning: 'perspective_factor_f' not explicitly provided. Dynamic 'f' calculation based on each mango is complex.")
            print("Assuming a pre-calibrated 'f' should be in perspective_config.")


    def extract_features(self, bbox_w_pixels, bbox_h_pixels):
        """
        Extracts features [a', b', e] for the XGBoost regression model.
        
        Args:
            bbox_w_pixels (float or np.ndarray): Width of the detected bounding box in pixels.
            bbox_h_pixels (float or np.ndarray): Height of the detected bounding box in pixels.
            
        Returns:
            dict: Dictionary of features {'a_prime_cm': val, 'b_prime_cm': val, 'eccentricity': val}
                  Returns None if perspective factor 'f' is not available.
        """
        if self.perspective_factor_f is None:
            # Try to get it from config if not set during init (e.g. from regression.yaml)
            # This is just a fallback, better to load it once.
            self.perspective_factor_f = self.config.get('perspective_factor_f') 
            if self.perspective_factor_f is None:
                 print("Error: Perspective factor 'f' is missing. Cannot extract features.")
                 # Handle this case: maybe return NaNs or raise error
                 return {'a_prime_cm': np.nan, 'b_prime_cm': np.nan, 'eccentricity': np.nan}


        # 1. Calculate initial ellipse axes (a, b) in pixels from bounding box
        a_px, b_px = calculate_ellipse_axes_from_bbox(bbox_w_pixels, bbox_h_pixels)
        
        # 2. Apply perspective correction to get a', b' (now in cm)
        #    The factor 'f' itself must be correctly calibrated to give cm/pixel.
        a_prime_cm, b_prime_cm = apply_perspective_correction(a_px, b_px, self.perspective_factor_f)
        
        # 3. Calculate eccentricity 'e' using the corrected axes a' and b' (or original a_px, b_px)
        #    Eccentricity is dimensionless, so it's the same whether calculated from (a,b) or (a',b').
        #    Let's use a_px, b_px as eccentricity is a shape property independent of scale.
        eccentricity = calculate_ellipse_eccentricity(a_px, b_px)
        
        return {
            'a_prime_cm': a_prime_cm, 
            'b_prime_cm': b_prime_cm, 
            'eccentricity': eccentricity
        }

if __name__ == '__main__':
    # Example perspective configuration (assuming 'f' is pre-calibrated)
    # This 'f' implies 1 pixel = 0.1 cm at the 5m distance.
    # This needs to be carefully determined for the specific camera setup.
    persp_config_example = {'perspective_factor_f': 0.1} 
    
    feature_extractor = SizeFeatureExtractor(perspective_config=persp_config_example)
    
    # Example detected bounding box dimensions
    detected_w_px = 120.0
    detected_h_px = 100.0
    
    features = feature_extractor.extract_features(detected_w_px, detected_h_px)
    
    if features:
        print(f"Extracted features for bbox {detected_w_px}x{detected_h_px}:")
        print(f"  a'_cm: {features['a_prime_cm']:.2f}")
        print(f"  b'_cm: {features['b_prime_cm']:.2f}")
        print(f"  Eccentricity: {features['eccentricity']:.4f}")

    # Example with a different perspective factor
    persp_config_example_2 = {'perspective_factor_f': 0.05} # Mangoes appear smaller or 'f' is different
    feature_extractor_2 = SizeFeatureExtractor(persp_config_example_2)
    features_2 = feature_extractor_2.extract_features(detected_w_px, detected_h_px)
    if features_2:
        print(f"\nExtracted features with f=0.05:")
        print(f"  a'_cm: {features_2['a_prime_cm']:.2f}")
        print(f"  b'_cm: {features_2['b_prime_cm']:.2f}")

    # Test case where 'f' might be missing (should print warning from init or error from extract)
    # persp_config_missing_f = {}
    # feature_extractor_missing_f = SizeFeatureExtractor(persp_config_missing_f)
    # features_missing_f = feature_extractor_missing_f.extract_features(detected_w_px, detected_h_px)
    # print(f"\nFeatures with missing 'f': {features_missing_f}")
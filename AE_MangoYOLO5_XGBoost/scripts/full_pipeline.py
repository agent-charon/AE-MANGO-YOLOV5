import torch
import cv2
import yaml
import os
import numpy as np
import pandas as pd

# Detection components
# from model.detection.detect_model import AEMangoYOLODetector
# from scripts.infer_detection import infer_single_image # Reuse inference logic

# Size estimation components
from model.size_estimation.feature_extractor import SizeFeatureExtractor
from model.size_estimation.xgboost_regressor import XGBoostMangoSizer
from model.size_estimation.ellipse_utils import calculate_ellipse_area


def run_full_pipeline(image_path, detection_model, detection_device, model_cfg_detect, 
                      nms_conf, nms_iou,
                      size_regressor, persp_config_reg,
                      output_image_path=None):
    """
    Runs the full pipeline: mango detection -> size estimation.
    Args:
        image_path (str): Path to the input image.
        detection_model: Trained AEMangoYOLODetector instance.
        detection_device: Device for detection model ('cpu' or 'cuda').
        model_cfg_detect: Config dict for detection (e.g. from model.yaml for UDIoU NMS params).
        nms_conf, nms_iou: NMS thresholds.
        size_regressor: Trained XGBoostMangoSizer instance.
        persp_config_reg: Perspective config for SizeFeatureExtractor.
        output_image_path (str, optional): Path to save image with detections and size info.
    Returns:
        list: List of dictionaries, each containing detection and size info for a mango.
              [{'bbox_xyxy': [x1,y1,x2,y2], 'confidence': conf, 
                'pixel_w': w_px, 'pixel_h': h_px,
                'est_major_axis_cm': a_cm, 'est_minor_axis_cm': b_cm, 
                'est_area_cm2': area_cm2}, ...]
    """
    print(f"\nProcessing image: {image_path}")
    
    # --- 1. Mango Detection ---
    # Using infer_single_image from scripts.infer_detection
    from scripts.infer_detection import infer_single_image, plot_one_box # Relative import
    
    # detections_scaled: list of [x1,y1,x2,y2,conf,cls_id, w_px,h_px]
    # img_with_boxes: image with plotted detections
    detections_scaled, img_display = infer_single_image(
        image_path, detection_model, detection_device, model_cfg_detect, 
        nms_conf, nms_iou, 
        output_image_path=None # We'll draw on img_display later
    )

    if not detections_scaled:
        print("No mangoes detected.")
        if output_image_path and img_display is not None:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, img_display) # Save original or image with no boxes
        return []

    print(f"Detected {len(detections_scaled)} mango(es).")

    # --- 2. Size Estimation for each detected mango ---
    feature_extractor = SizeFeatureExtractor(perspective_config=persp_config_reg)
    results = []

    for i, det in enumerate(detections_scaled):
        # det is [x1, y1, x2, y2, conf, cls_id, w_px, h_px]
        bbox_xyxy = det[:4]
        confidence = det[4]
        # class_id = det[5] # Assuming class 0 is mango
        pixel_w = det[6]
        pixel_h = det[7]

        print(f"  Mango {i+1}: BBox Pixels W={pixel_w:.1f}, H={pixel_h:.1f}, Conf={confidence:.2f}")

        if pixel_w <=0 or pixel_h <=0:
            print(f"  Skipping Mango {i+1} due to invalid pixel dimensions (w={pixel_w}, h={pixel_h})")
            continue

        # Extract features [a'_cm, b'_cm, eccentricity]
        # SizeFeatureExtractor expects bbox_w_pixels, bbox_h_pixels
        features_dict = feature_extractor.extract_features(pixel_w, pixel_h)
        
        if np.isnan(features_dict['a_prime_cm']): # Check if feature extraction failed
            print(f"  Skipping Mango {i+1} due to feature extraction error (e.g. missing perspective factor).")
            continue
            
        # Prepare feature vector for XGBoost
        # Order: ['a_prime_cm', 'b_prime_cm', 'eccentricity']
        xgb_features = np.array([[
            features_dict['a_prime_cm'],
            features_dict['b_prime_cm'],
            features_dict['eccentricity']
        ]])

        # Predict actual_a_cm, actual_b_cm
        predicted_axes_cm = size_regressor.predict(xgb_features) # Shape (1, 2)
        est_a_cm = predicted_axes_cm[0, 0]
        est_b_cm = predicted_axes_cm[0, 1]
        
        # Calculate estimated area
        est_area_cm2 = calculate_ellipse_area(est_a_cm, est_b_cm)

        print(f"    Estimated: a={est_a_cm:.2f}cm, b={est_b_cm:.2f}cm, Area={est_area_cm2:.2f}cm²")
        
        results.append({
            'bbox_xyxy': bbox_xyxy,
            'confidence': confidence,
            'pixel_w': pixel_w,
            'pixel_h': pixel_h,
            'input_features_to_xgb': xgb_features.flatten().tolist(),
            'est_major_axis_cm': est_a_cm,
            'est_minor_axis_cm': est_b_cm,
            'est_area_cm2': est_area_cm2
        })

        # Add size info to the display image
        label_size = f"Area:{est_area_cm2:.1f}cm2"
        # Position text below existing label or near bbox
        text_x = int(bbox_xyxy[0])
        text_y = int(bbox_xyxy[3]) + 20 # Below bottom-left y
        if text_y > img_display.shape[0] - 10 : text_y = int(bbox_xyxy[1]) - 5 # Above top-left y
        
        cv2.putText(img_display, label_size, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

    if output_image_path and img_display is not None:
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        cv2.imwrite(output_image_path, img_display)
        print(f"Saved full pipeline result to: {output_image_path}")
        
    return results


def main():
    # --- Load All Configs ---
    with open("configs/model.yaml", 'r') as f: # Detection model NMS params, etc.
        model_cfg_detect = yaml.safe_load(f)
    with open("configs/training.yaml", 'r') as f: # Device, checkpoint paths
        training_cfg = yaml.safe_load(f)
    with open("configs/regression.yaml", 'r') as f: # Perspective factor, XGBoost params
        regression_cfg = yaml.safe_load(f)
    # dataset_cfg might be needed if perspective factor 'f' depends on it.

    detection_device = torch.device(training_cfg['device'] if torch.cuda.is_available() else "cpu")

    # --- Load Detection Model ---
    from model.detection.detect_model import AEMangoYOLODetector # Import here
    detection_model = AEMangoYOLODetector(cfg_model_yaml_path="configs/model.yaml", num_classes_override=1)
    
    det_checkpoint_path = os.path.join(training_cfg['detection_checkpoint_dir'], "ae_mangoyolo5_epoch_20.pt") # Example
    if not os.path.exists(det_checkpoint_path):
        print(f"ERROR: Detection model checkpoint not found at {det_checkpoint_path}.")
        # Fallback to latest if exists
        import glob
        checkpoints = sorted(glob.glob(os.path.join(training_cfg['detection_checkpoint_dir'], "*.pt")))
        if checkpoints: det_checkpoint_path = checkpoints[-1]; print(f"Using latest: {det_checkpoint_path}")
        else: return
    try:
        ckpt_det = torch.load(det_checkpoint_path, map_location=detection_device)
        detection_model.load_state_dict(ckpt_det['model_state_dict'])
        detection_model.to(detection_device).eval()
        print(f"Loaded detection model from {det_checkpoint_path}")
    except Exception as e:
        print(f"Error loading detection model: {e}"); return


    # --- Load Size Estimation Regressor ---
    reg_model_path = training_cfg.get('regression_model_save_path', "outputs/models/xgboost_mango_sizer.json")
    if not os.path.exists(reg_model_path):
        print(f"ERROR: Size regressor model not found at {reg_model_path}.")
        return
    
    size_regressor = XGBoostMangoSizer(regression_cfg, training_cfg)
    size_regressor.load_model(reg_model_path)
    print(f"Loaded size regressor model from {reg_model_path}")


    # --- Prepare Perspective Config for Feature Extractor ---
    persp_factor_f = regression_cfg.get('perspective_factor_f')
    if persp_factor_f is None:
        print("ERROR: 'perspective_factor_f' is crucial and not found in regression.yaml.")
        return
    persp_config_reg = {'perspective_factor_f': persp_factor_f}


    # --- NMS Parameters for Detection ---
    nms_conf = model_cfg_detect.get('nms_conf_threshold', 0.25)
    nms_iou = model_cfg_detect.get('nms_iou_threshold', 0.45) # UDIoU comparison threshold

    # --- Input Image / Directory ---
    input_image_file = "path/to/your/single_test_mango_image.jpg" # Replace
    output_dir_pipeline = "outputs/full_pipeline_results/"
    os.makedirs(output_dir_pipeline, exist_ok=True)
    
    if not os.path.exists(input_image_file):
        print(f"ERROR: Test image not found: {input_image_file}. Please provide a valid image.")
        # Create a dummy image for testing if none provided
        dummy_img = np.zeros((480,640,3), dtype=np.uint8)
        cv2.circle(dummy_img, (320,240), 50, (0,180,0), -1) # A green "mango"
        input_image_file = os.path.join(output_dir_pipeline, "dummy_mango_test_img.png")
        cv2.imwrite(input_image_file, dummy_img)
        print(f"Created a dummy test image at {input_image_file}")


    output_image_annotated_path = os.path.join(output_dir_pipeline, os.path.basename(input_image_file))

    pipeline_results = run_full_pipeline(
        input_image_file, detection_model, detection_device, model_cfg_detect,
        nms_conf, nms_iou,
        size_regressor, persp_config_reg,
        output_image_path=output_image_annotated_path
    )

    print("\n--- Full Pipeline Results ---")
    if pipeline_results:
        df_results = pd.DataFrame(pipeline_results)
        print(df_results[['pixel_w', 'pixel_h', 'est_major_axis_cm', 'est_minor_axis_cm', 'est_area_cm2', 'confidence']])
        results_csv_path = os.path.join(output_dir_pipeline, "pipeline_summary.csv")
        df_results.to_csv(results_csv_path, index=False)
        print(f"Saved summary results to {results_csv_path}")
    else:
        print("Pipeline did not produce any results.")

if __name__ == '__main__':
    print("NOTE: Full pipeline script. Requires trained detection and regression models.")
    print("Replace 'path/to/your/single_test_mango_image.jpg' with an actual image.")
    try:
        main()
    except FileNotFoundError as e:
        print(f"Missing file or directory: {e}. Please check paths and configs.")
    except Exception as e:
        print(f"An error occurred during full pipeline execution: {e}")
        import traceback
        traceback.print_exc()
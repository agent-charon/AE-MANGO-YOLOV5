import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os

from model.size_estimation.feature_extractor import SizeFeatureExtractor
from model.size_estimation.xgboost_regressor import XGBoostMangoSizer
from model.size_estimation.ellipse_utils import calculate_ellipse_axes_from_bbox # For GT a,b if needed

def load_data_for_regression(dataset_cfg, regression_cfg):
    """
    Loads data needed for training the size estimation regressor.
    This involves:
    1. Detected bounding boxes (from a detection model or ground truth annotations).
       For training XGBoost, we need (bbox_w_pixels, bbox_h_pixels) for each mango.
    2. Ground truth actual sizes (a_cm, b_cm) from caliper_loader.
    3. (Optional) Bounding boxes for the Lpos DIoU term.
    """
    print("Loading data for regression training...")
    # For this script, we need a dataset that pairs:
    # (detected_bbox_w_px, detected_bbox_h_px) -> features [a'_cm, b'_cm, e]
    # with ground_truth [actual_a_cm, actual_b_cm]

    # Option 1: Use ground truth annotations for bounding boxes to train regression initially.
    # This decouples detection errors from regression training.
    # The paper uses "detected bounding box parameters", suggesting output from detector.
    
    # For now, let's assume we have a CSV file that already contains:
    # mango_id, detected_bbox_w_px, detected_bbox_h_px, gt_actual_width_cm, gt_actual_height_cm
    # (And optionally, pred_bbox_cx, pred_bbox_cy, pred_bbox_w, pred_bbox_h, 
    #  gt_bbox_cx, gt_bbox_cy, gt_bbox_w, gt_bbox_h for Lpos DIoU)

    # --- Placeholder for data loading ---
    # This needs to be replaced with actual data source.
    # Paper: "200 mangoes were selected for size estimation task based on their visibility"
    # "Width and height were precisely measured with a Vernier Caliper"
    # "Bounding box parameters ... are used to correlate with actual mango dimensions"
    
    num_samples = dataset_cfg.get('size_estimation_total_samples', 200)
    print(f"Using {num_samples} samples for size estimation based on dataset_cfg.")

    # Mock data generation:
    # Features: detected_bbox_w_px, detected_bbox_h_px
    # Targets: gt_actual_major_axis_cm, gt_actual_minor_axis_cm
    # BBoxes for Lpos: pred_bbox_xywh, gt_bbox_xywh (pixels)
    
    # Let's assume gt_actual_width_cm and gt_actual_height_cm from caliper are for the mango's max dimensions.
    # We need to convert these to gt_actual_major_axis_cm and gt_actual_minor_axis_cm of the *ellipse model*
    # that the regressor is trying to predict.
    # If the regressor predicts a_ellipse, b_ellipse, then GT should be a_ellipse_gt, b_ellipse_gt.
    # The paper's Lpos uses (a,b) and (agt,bgt). These agt,bgt should be ellipse axes.
    
    # Mock 'detected' bounding box dimensions in pixels
    # Mangoes in pixels might be e.g. 50-150 pixels depending on distance/resolution
    mock_detected_w_px = np.random.uniform(80, 150, num_samples)
    mock_detected_h_px = mock_detected_w_px * np.random.uniform(0.7, 1.3, num_samples) # Varying aspect ratios
    
    # Mock ground truth actual ellipse axes in cm (what we want to predict)
    # These would come from caliper measurements, possibly converted to ellipse axes.
    # If caliper gives W_mango, H_mango, we can use calculate_ellipse_axes_from_bbox(W_mango, H_mango)
    # to get a_gt_cm, b_gt_cm, assuming the caliper measurements define an "actual bounding box" in cm.
    mock_gt_width_cm = np.random.uniform(7, 12, num_samples) # Caliper width
    mock_gt_height_cm = mock_gt_width_cm * np.random.uniform(0.8, 1.2, num_samples) # Caliper height/length
    
    gt_a_cm_ellipse, gt_b_cm_ellipse = calculate_ellipse_axes_from_bbox(mock_gt_width_cm, mock_gt_height_cm)
    
    y_data = np.stack([gt_a_cm_ellipse, gt_b_cm_ellipse], axis=1)

    # Mock bounding boxes in pixels for Lpos DIoU term (if use_custom_lpos is True)
    # These would be the bounding boxes from which mock_detected_w_px, mock_detected_h_px were derived.
    # pred_bbox_xywh: [cx, cy, w, h] corresponding to mock_detected_w_px, mock_detected_h_px
    # gt_bbox_xywh: ground truth bounding boxes for the same mangoes (e.g., manual annotations)
    mock_pred_bbox_xywh = np.zeros((num_samples, 4))
    mock_pred_bbox_xywh[:, 0] = np.random.uniform(200, 800, num_samples) # cx
    mock_pred_bbox_xywh[:, 1] = np.random.uniform(200, 800, num_samples) # cy
    mock_pred_bbox_xywh[:, 2] = mock_detected_w_px
    mock_pred_bbox_xywh[:, 3] = mock_detected_h_px
    
    # GT bboxes slightly different from pred_bboxes
    mock_gt_bbox_xywh = mock_pred_bbox_xywh.copy()
    mock_gt_bbox_xywh[:, :2] += np.random.uniform(-5, 5, (num_samples, 2)) # Shift centers
    mock_gt_bbox_xywh[:, 2:] += np.random.uniform(-5, 5, (num_samples, 2)) # Adjust w,h
    mock_gt_bbox_xywh[:, 2:] = np.maximum(mock_gt_bbox_xywh[:, 2:], 10) # Ensure positive w,h


    # --- Feature Extraction ---
    # This perspective_factor_f needs to be calibrated for your setup.
    # It converts pixel dimensions (a_px, b_px) from bounding box to cm (a_prime_cm, b_prime_cm).
    # The paper uses a fixed 5m distance.
    # Example: If at 5m, an object of 10cm height is 100 pixels, then f for height is 10/100 = 0.1 cm/pixel.
    # This 'f' needs to be consistent.
    persp_factor = regression_cfg.get('perspective_factor_f', 0.1) # Example value
    persp_config = {'perspective_factor_f': persp_factor}
    extractor = SizeFeatureExtractor(perspective_config=persp_config)

    X_data_list = []
    for w_px, h_px in zip(mock_detected_w_px, mock_detected_h_px):
        features = extractor.extract_features(w_px, h_px)
        X_data_list.append([features['a_prime_cm'], features['b_prime_cm'], features['eccentricity']])
    X_data = np.array(X_data_list)
    
    # Remove samples with NaN features (e.g. if perspective factor was bad)
    nan_mask = np.isnan(X_data).any(axis=1)
    if np.any(nan_mask):
        print(f"Warning: Removing {np.sum(nan_mask)} samples with NaN features.")
        X_data = X_data[~nan_mask]
        y_data = y_data[~nan_mask]
        mock_pred_bbox_xywh = mock_pred_bbox_xywh[~nan_mask]
        mock_gt_bbox_xywh = mock_gt_bbox_xywh[~nan_mask]

    print(f"Generated {X_data.shape[0]} samples for regression.")
    print(f"X_data shape: {X_data.shape}, y_data shape: {y_data.shape}")
    print(f"Sample X: {X_data[0]}, Sample y: {y_data[0]}")

    return X_data, y_data, mock_pred_bbox_xywh, mock_gt_bbox_xywh


def main():
    with open("configs/dataset.yaml", 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    with open("configs/regression.yaml", 'r') as f:
        regression_cfg = yaml.safe_load(f)
    with open("configs/training.yaml", 'r') as f:
        training_cfg = yaml.safe_load(f)

    # Load data
    X, y, pred_bboxes, gt_bboxes = load_data_for_regression(dataset_cfg, regression_cfg)

    if X.shape[0] == 0:
        print("No data loaded for regression. Exiting.")
        return

    # Train/test split (paper uses 180 train, 20 test from 200 samples)
    # Calculate test_size based on config
    total_s = dataset_cfg.get('size_estimation_total_samples', 200)
    test_s = dataset_cfg.get('size_estimation_test_samples', 20)
    test_split_ratio = test_s / total_s if total_s > 0 else 0.1 # Default to 10% if not specified

    indices = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, indices, test_size=test_split_ratio, random_state=training_cfg['seed']
    )
    
    pred_bboxes_train = pred_bboxes[idx_train] if pred_bboxes is not None else None
    gt_bboxes_train = gt_bboxes[idx_train] if gt_bboxes is not None else None
    pred_bboxes_test = pred_bboxes[idx_test] if pred_bboxes is not None else None # For eval if Lpos affects it
    gt_bboxes_test = gt_bboxes[idx_test] if gt_bboxes is not None else None


    print(f"Training data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    print(f"Test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")

    # Initialize and train model
    regressor = XGBoostMangoSizer(regression_cfg, training_cfg)
    
    print("Starting regression model training...")
    regressor.train(X_train, y_train, X_test, y_test,
                    train_pred_bboxes=pred_bboxes_train if regression_cfg.get('use_custom_lpos') else None,
                    train_gt_bboxes=gt_bboxes_train if regression_cfg.get('use_custom_lpos') else None,
                    # Eval bboxes are not directly used by standard XGBoost obj/eval during training,
                    # but could be if a custom eval metric needed them.
                    eval_pred_bboxes=pred_bboxes_test if regression_cfg.get('use_custom_lpos') else None,
                    eval_gt_bboxes=gt_bboxes_test if regression_cfg.get('use_custom_lpos') else None
                    )

    # Evaluate
    y_pred_test = regressor.predict(X_test)
    
    r2_a = r2_score(y_test[:, 0], y_pred_test[:, 0])
    mae_a = mean_absolute_error(y_test[:, 0], y_pred_test[:, 0])
    r2_b = r2_score(y_test[:, 1], y_pred_test[:, 1])
    mae_b = mean_absolute_error(y_test[:, 1], y_pred_test[:, 1])

    print("\n--- Test Set Evaluation ---")
    print(f"R^2 Score (major axis a): {r2_a:.4f}")
    print(f"MAE (major axis a): {mae_a:.4f} cm")
    print(f"R^2 Score (minor axis b): {r2_b:.4f}")
    print(f"MAE (minor axis b): {mae_b:.4f} cm")

    # Area calculation and R2 for area (as in paper R2=0.91)
    # Area_pred = pi * pred_a * pred_b
    # Area_gt = pi * gt_a * gt_b
    pred_area = np.pi * y_pred_test[:, 0] * y_pred_test[:, 1]
    gt_area = np.pi * y_test[:, 0] * y_test[:, 1]
    
    r2_area = r2_score(gt_area, pred_area)
    mae_area = mean_absolute_error(gt_area, pred_area)
    print(f"R^2 Score (Ellipse Area): {r2_area:.4f}") # Compare this to paper's 0.91
    print(f"MAE (Ellipse Area): {mae_area:.4f} cm^2") # Compare this to paper's 2.8 cm^2

    # Save model
    model_save_path = training_cfg.get('regression_model_save_path', "outputs/models/xgboost_mango_sizer.json")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    regressor.save_model(model_save_path)
    print(f"Regression model saved to {model_save_path}")
    
    print("Regression model training finished.")

if __name__ == '__main__':
    # Create dummy config files if they don't exist for a quick test run
    if not os.path.exists("configs/dataset.yaml"): print("WARNING: configs/dataset.yaml not found.")
    if not os.path.exists("configs/regression.yaml"): print("WARNING: configs/regression.yaml not found.")
    if not os.path.exists("configs/training.yaml"): print("WARNING: configs/training.yaml not found.")
    
    # os.makedirs("outputs/models", exist_ok=True) # Ensure output dir exists

    try:
        main()
    except FileNotFoundError as e:
        print(f"Missing config file: {e}. Please ensure all YAML configs are present.")
    except Exception as e:
        print(f"An error occurred during regression training script: {e}")
        import traceback
        traceback.print_exc()
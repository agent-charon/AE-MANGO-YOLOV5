import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import os

from model.size_estimation.feature_extractor import SizeFeatureExtractor
from model.size_estimation.xgboost_regressor import XGBoostMangoSizer
from scripts.train_regression import load_data_for_regression # Reuse data loading for consistency

def evaluate_regressor(regressor_model, X_data, y_data, model_name="XGBoost"):
    """
    Evaluates a trained regressor.
    Args:
        regressor_model: Trained regressor instance (e.g., XGBoostMangoSizer).
        X_data (np.ndarray): Feature data [a_prime, b_prime, eccentricity].
        y_data (np.ndarray): True target data [actual_a_cm, actual_b_cm].
        model_name (str): Name of the model for printing results.
    Returns:
        dict: Dictionary of metrics.
    """
    if X_data.shape[0] == 0:
        print(f"No data to evaluate for {model_name}.")
        return None

    y_pred = regressor_model.predict(X_data) # Predicts [pred_a_cm, pred_b_cm]

    gt_a_cm = y_data[:, 0]
    pred_a_cm = y_pred[:, 0]
    gt_b_cm = y_data[:, 1]
    pred_b_cm = y_pred[:, 1]

    r2_a = r2_score(gt_a_cm, pred_a_cm)
    mae_a = mean_absolute_error(gt_a_cm, pred_a_cm)
    r2_b = r2_score(gt_b_cm, pred_b_cm)
    mae_b = mean_absolute_error(gt_b_cm, pred_b_cm)

    # Area calculation
    pred_area = np.pi * pred_a_cm * pred_b_cm
    gt_area = np.pi * gt_a_cm * gt_b_cm
    
    r2_area_val = r2_score(gt_area, pred_area)
    mae_area_val = mean_absolute_error(gt_area, pred_area)
    
    print(f"\n--- Evaluation Results for {model_name} ---")
    print(f"  R^2 Score (major axis a): {r2_a:.4f}")
    print(f"  MAE (major axis a): {mae_a:.4f} cm")
    print(f"  R^2 Score (minor axis b): {r2_b:.4f}")
    print(f"  MAE (minor axis b): {mae_b:.4f} cm")
    print(f"  R^2 Score (Ellipse Area): {r2_area_val:.4f}")
    print(f"  MAE (Ellipse Area): {mae_area_val:.4f} cm^2")

    metrics = {
        'r2_a': r2_a, 'mae_a': mae_a,
        'r2_b': r2_b, 'mae_b': mae_b,
        'r2_area': r2_area_val, 'mae_area': mae_area_val,
        'gt_area': gt_area, 'pred_area': pred_area
    }
    return metrics

def plot_regression_results(gt_values, pred_values, title, xlabel, ylabel, r2_val, output_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(gt_values, pred_values, alpha=0.7, edgecolors='k', label=f'R² = {r2_val:.2f}')
    
    # Line of equality
    min_val = min(np.min(gt_values), np.min(pred_values))
    max_val = max(np.max(gt_values), np.max(pred_values))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Equality Line')
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot: {output_path}")


def main():
    with open("configs/dataset.yaml", 'r') as f:
        dataset_cfg = yaml.safe_load(f)
    with open("configs/regression.yaml", 'r') as f:
        regression_cfg = yaml.safe_load(f)
    with open("configs/training.yaml", 'r') as f:
        training_cfg = yaml.safe_load(f)

    # Load data (using the same function as training to get consistent test set)
    X_all, y_all, _, _ = load_data_for_regression(dataset_cfg, regression_cfg)
    
    if X_all.shape[0] == 0:
        print("No data loaded for evaluation. Exiting.")
        return

    # Split into train/test to get the test set
    # Note: This assumes evaluate_regression is run after train_regression,
    # or that the split here is consistent with how the model was trained.
    total_s = dataset_cfg.get('size_estimation_total_samples', 200)
    test_s = dataset_cfg.get('size_estimation_test_samples', 20)
    test_split_ratio = test_s / total_s if total_s > 0 else 0.1

    from sklearn.model_selection import train_test_split # Already imported but good for clarity
    _, X_test, _, y_test = train_test_split(
        X_all, y_all, test_size=test_split_ratio, random_state=training_cfg['seed']
    )
    print(f"Using {X_test.shape[0]} samples for evaluation (test set).")


    # Load the trained XGBoost model
    model_load_path = training_cfg.get('regression_model_save_path', "outputs/models/xgboost_mango_sizer.json")
    if not os.path.exists(model_load_path):
        print(f"Error: Trained regression model not found at {model_load_path}. Train the model first.")
        return
        
    regressor = XGBoostMangoSizer(regression_cfg, training_cfg) # Init with same configs
    regressor.load_model(model_load_path)
    print(f"Loaded regression model from {model_load_path}")

    # Evaluate
    model_name = "XGBoost_with_Lpos" if regression_cfg.get('use_custom_lpos') else "XGBoost_standard"
    eval_metrics = evaluate_regressor(regressor, X_test, y_test, model_name=model_name)

    # Plotting (like Fig. 8 in paper for area)
    if eval_metrics:
        output_plot_dir = "outputs/results/regression_plots/"
        os.makedirs(output_plot_dir, exist_ok=True)

        plot_regression_results(
            gt_values=eval_metrics['gt_area'],
            pred_values=eval_metrics['pred_area'],
            title=f'{model_name} - Actual vs. Estimated Mango Area',
            xlabel='Actual Area (cm²)',
            ylabel='Estimated Area (cm²)',
            r2_val=eval_metrics['r2_area'],
            output_path=os.path.join(output_plot_dir, f"{model_name.lower()}_area_fit.png")
        )
        
        # Plot distribution (like Fig. 9)
        plt.figure(figsize=(10, 6))
        plt.hist(eval_metrics['gt_area'], bins=15, alpha=0.7, label='Actual Area', color='black', density=True)
        plt.hist(eval_metrics['pred_area'], bins=15, alpha=0.7, label='Estimated Area', color='red', density=True)
        plt.title(f'{model_name} - Distribution of Actual vs. Estimated Mango Area')
        plt.xlabel('Area (cm²)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True)
        dist_plot_path = os.path.join(output_plot_dir, f"{model_name.lower()}_area_distribution.png")
        plt.savefig(dist_plot_path)
        plt.close()
        print(f"Saved distribution plot: {dist_plot_path}")


if __name__ == '__main__':
    # Ensure configs and trained model exist
    print("NOTE: Regression evaluation script. Assumes a trained regression model exists.")
    try:
        main()
    except FileNotFoundError as e:
        print(f"Missing file or directory: {e}. Please check paths and configs.")
    except Exception as e:
        print(f"An error occurred during regression evaluation: {e}")
        import traceback
        traceback.print_exc()
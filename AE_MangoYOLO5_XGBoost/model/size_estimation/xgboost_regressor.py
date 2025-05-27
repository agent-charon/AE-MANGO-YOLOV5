import xgboost as xgb
import numpy as np
import json
from .positional_loss import positional_loss_xgboost # Custom loss (with placeholder grad/hess)
from functools import partial

class XGBoostMangoSizer:
    def __init__(self, regression_config, training_config):
        self.params = {
            'objective': regression_config.get('objective', 'reg:squarederror'), # Default if no custom
            'eval_metric': ['rmse', 'mae'], # Standard metrics
            'eta': regression_config.get('learning_rate', 0.05), # learning_rate
            'max_depth': regression_config.get('max_depth', 5),
            'subsample': regression_config.get('subsample', 0.8),
            'colsample_bytree': regression_config.get('colsample_bytree', 0.8),
            'gamma': regression_config.get('gamma', 0.1),
            'alpha': regression_config.get('reg_alpha', 0.1), # L1
            'lambda': regression_config.get('reg_lambda', 0.1), # L2
            'nthread': -1, # Use all available threads
            'seed': training_config.get('seed', 42)
        }
        self.model = None
        self.num_boost_round = regression_config.get('n_estimators', 200)
        self.early_stopping_rounds = training_config.get('xgboost_early_stopping_rounds', 20)
        self.loss_coeffs = {
            'alpha_lpos': regression_config.get('alpha_lpos', 0.1),
            'beta_lpos': regression_config.get('beta_lpos', 0.1),
            'gamma_lpos': regression_config.get('gamma_lpos', 0.1),
            'delta_lpos': regression_config.get('delta_lpos', 0.1)
        }
        self.use_custom_lpos = regression_config.get('use_custom_lpos', False) # Control Lpos usage

        # Store bounding boxes if Lpos needs them. This is tricky for XGBoost's DMatrix.
        # Typically, features for DMatrix are only from X_train.
        # One way is to make bboxes part of X_train if Lpos is dynamic.
        # Or, if Lpos DIoU term is based on relatively fixed bboxes per sample.
        self.train_pred_bboxes = None
        self.train_gt_bboxes = None
        self.eval_pred_bboxes = None
        self.eval_gt_bboxes = None


    def train(self, X_train, y_train, X_val=None, y_val=None, 
              train_pred_bboxes=None, train_gt_bboxes=None,
              eval_pred_bboxes=None, eval_gt_bboxes=None):
        """
        Train the XGBoost regressor.
        X_train: features [a_prime_cm, b_prime_cm, eccentricity]
        y_train: target [actual_a_cm, actual_b_cm]
        *_bboxes: Optional bounding boxes for Lpos DIoU term.
                  These should correspond to the samples in X_train/X_val.
        """
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
        else:
            evals = [(dtrain, 'train')]

        # Store bboxes for custom loss function if provided
        # This is a bit of a hack, as obj function only takes (preds, dtrain)
        # The bboxes should ideally be part of dtrain or globally accessible.
        # A cleaner way for complex inputs to loss: use XGBoost's Python API with custom data iterators.
        # For now, we use partial to pass them.
        self.train_pred_bboxes = train_pred_bboxes
        self.train_gt_bboxes = train_gt_bboxes
        self.eval_pred_bboxes = eval_pred_bboxes # Not used by obj during training, but for consistency
        self.eval_gt_bboxes = eval_gt_bboxes
        
        obj_func = None
        if self.use_custom_lpos:
            print("Using custom Lpos for XGBoost training.")
            # Partial function to include loss_coeffs and bboxes for the training set
            # Note: This only works if the `dtrain` object passed to obj_func during
            # XGBoost's internal training loop is the *same instance* as `dtrain` here,
            # or if we can somehow map samples in `dtrain` to their bboxes.
            # Standard XGBoost `obj(preds, dtrain)` means `dtrain` is the current batch/set.
            # We will assume bboxes are passed for the *entire* dtrain set.
            # This requires positional_loss_xgboost to handle full batch.
            obj_func = partial(positional_loss_xgboost, 
                               loss_coeffs=self.loss_coeffs,
                               pred_bbox_xywh=self.train_pred_bboxes, # These are for the *training* set
                               gt_bbox_xywh=self.train_gt_bboxes)
            # Clear default objective if custom is used
            if 'objective' in self.params and self.params['objective'] != 'reg:squarederror': # if custom was set
                 pass # obj_func will override
            else: # if default was squarederror
                 self.params.pop('objective', None) # Remove it so obj_func is primary

        print(f"XGBoost params: {self.params}")
        if obj_func:
            print(f"Using custom objective: {obj_func}")
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            obj=obj_func if self.use_custom_lpos else None, # Pass custom objective
            # feval=custom_eval_metric if needed
            verbose_eval=50
        )

    def predict(self, X_test):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        dtest = xgb.DMatrix(X_test)
        return self.model.predict(dtest)

    def save_model(self, filepath):
        if self.model is None:
            # print("Warning: No model to save.")
            return
        self.model.save_model(filepath)
        # Save loss_coeffs as well, as they are part of the Lpos logic
        # (though not part of XGB model file itself)
        meta_filepath = filepath.replace(".json", "_meta.json")
        with open(meta_filepath, 'w') as f:
            json.dump({'loss_coeffs': self.loss_coeffs, 'use_custom_lpos': self.use_custom_lpos}, f)


    def load_model(self, filepath):
        self.model = xgb.Booster()
        self.model.load_model(filepath)
        meta_filepath = filepath.replace(".json", "_meta.json")
        try:
            with open(meta_filepath, 'r') as f:
                meta = json.load(f)
                self.loss_coeffs = meta.get('loss_coeffs', self.loss_coeffs)
                self.use_custom_lpos = meta.get('use_custom_lpos', self.use_custom_lpos)
        except FileNotFoundError:
            print(f"Meta file {meta_filepath} not found. Using default Lpos settings.")


if __name__ == '__main__':
    # Example usage:
    num_train_samples = 180
    num_val_samples = 20
    num_features = 3 # a_prime, b_prime, eccentricity

    # Mock data
    X_tr = np.random.rand(num_train_samples, num_features)
    y_tr = np.random.rand(num_train_samples, 2) * 10 # target: actual_a, actual_b

    X_v = np.random.rand(num_val_samples, num_features)
    y_v = np.random.rand(num_val_samples, 2) * 10
    
    # Mock bboxes for Lpos (optional)
    train_pred_bb = np.random.rand(num_train_samples, 4) * 100 + 50
    train_gt_bb = train_pred_bb + (np.random.rand(num_train_samples, 4) - 0.5) * 5
    
    mock_reg_config = {
        'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50,
        'use_custom_lpos': True, # Set to True to test Lpos
        'alpha_lpos': 0.1, 'beta_lpos': 0.2, 'gamma_lpos': 0.1, 'delta_lpos': 0.2
    }
    mock_train_config = {'seed': 42, 'xgboost_early_stopping_rounds': 10}

    sizer = XGBoostMangoSizer(mock_reg_config, mock_train_config)
    
    print("Training XGBoost sizer...")
    sizer.train(X_tr, y_tr, X_v, y_v, 
                train_pred_bboxes=train_pred_bb if mock_reg_config['use_custom_lpos'] else None,
                train_gt_bboxes=train_gt_bb if mock_reg_config['use_custom_lpos'] else None)
    
    print("\nPredicting with trained sizer:")
    X_test_ex = np.random.rand(5, num_features)
    predictions = sizer.predict(X_test_ex)
    print("Predictions (actual_a, actual_b):\n", predictions)

    model_path = "temp_xgb_sizer.json"
    sizer.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    sizer_loaded = XGBoostMangoSizer(mock_reg_config, mock_train_config)
    sizer_loaded.load_model(model_path)
    print("Model loaded.")
    predictions_loaded = sizer_loaded.predict(X_test_ex)
    assert np.allclose(predictions, predictions_loaded), "Loaded model prediction mismatch!"
    print("Loaded model predictions match.")

    # Clean up temp file
    import os
    if os.path.exists(model_path): os.remove(model_path)
    if os.path.exists(model_path.replace(".json", "_meta.json")): os.remove(model_path.replace(".json", "_meta.json"))
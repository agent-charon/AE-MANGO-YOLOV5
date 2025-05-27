import torch
import cv2
import yaml
import os
import numpy as np

# from model.detection.detect_model import AEMangoYOLODetector
# from model.detection.utils.nms import non_max_suppression_udiou, xywh2xyxy, xyxy2xywh
# from model.detection.utils.plots import plot_one_box # Utility to draw boxes

# --- Placeholder for plot_one_box ---
def plot_one_box(xyxy, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img

def infer_single_image(image_path, model, device, model_cfg, nms_conf_thres, nms_iou_thres, output_image_path=None):
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not read image {image_path}")
        return None, None

    # Preprocess image (resize, normalize, to tensor) - similar to training
    # Assuming model expects 640x640 input, or get from config
    img_size_infer = (640, 640) # Example, or from model config
    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (img_size_infer[1], img_size_infer[0])) # W, H
    img_tensor = torch.from_numpy(img_resized).permute(2,0,1).float().to(device) / 255.0
    img_tensor = img_tensor.unsqueeze(0) # Add batch dimension

    model.eval()
    with torch.no_grad():
        # predictions_infer: (tensor of detections [N, num_outputs], list of raw feature maps)
        processed_detections_batch, _ = model(img_tensor) 
    
    # Apply NMS (non_max_suppression_udiou expects [bs, num_queries, 5+nc])
    # processed_detections_batch is already [bs, total_dets, 5+nc] from Detect layer's inference path
    from model.detection.utils.nms import non_max_suppression_udiou
    udiou_params = { # Load these from model_cfg or use sensible defaults
        'v_aspect_term_coeff': model_cfg.get('v_aspect_term_coeff', 0.5),
        'lambda_scale_invariance_coeff': model_cfg.get('lambda_scale_invariance_coeff', 0.3),
        'beta_p_center_dist_coeff': model_cfg.get('beta_p_center_dist_coeff', 0.2),
        'gamma_theta_orientation_coeff': model_cfg.get('gamma_theta_orientation_coeff', 0.1)
    }
    
    # The output of non_max_suppression_udiou is a list (per image in batch)
    # Each item is a tensor [N_det, 6] where 6 is [x1, y1, x2, y2, conf, cls_idx]
    # Coordinates are relative to the model input size (e.g., 640x640)
    detections_after_nms_list = non_max_suppression_udiou(
        processed_detections_batch, 
        conf_thres=nms_conf_thres, 
        iou_thres=nms_iou_thres, # This threshold is for UDIoU comparison
        udiou_params=udiou_params,
        multi_label=False # Assuming single class (mango) or best class
    )
    
    detections_scaled = [] # Store scaled [x1,y1,x2,y2,conf,cls, w_px, h_px]
    
    # We processed one image, so take the first element
    if detections_after_nms_list and detections_after_nms_list[0].numel() > 0:
        detections_for_image = detections_after_nms_list[0] # Tensor [N_det, 6]
        
        # Scale boxes back to original image size
        orig_h, orig_w = img_orig.shape[:2]
        input_h, input_w = img_size_infer # Model input size (e.g., 640, 640)

        # Clone to avoid modifying NMS output directly if it's reused
        scaled_boxes = detections_for_image.clone()

        scaled_boxes[:, 0] = detections_for_image[:, 0] * (orig_w / input_w) # x1
        scaled_boxes[:, 1] = detections_for_image[:, 1] * (orig_h / input_h) # y1
        scaled_boxes[:, 2] = detections_for_image[:, 2] * (orig_w / input_w) # x2
        scaled_boxes[:, 3] = detections_for_image[:, 3] * (orig_h / input_h) # y2
        
        # Clamp boxes to image dimensions
        scaled_boxes[:, 0:4] = scaled_boxes[:, 0:4].round()
        scaled_boxes[:, 0].clamp_(0, orig_w)  # x1
        scaled_boxes[:, 1].clamp_(0, orig_h)  # y1
        scaled_boxes[:, 2].clamp_(0, orig_w)  # x2
        scaled_boxes[:, 3].clamp_(0, orig_h)  # y2

        img_with_boxes = img_orig.copy()
        for det in scaled_boxes:
            xyxy = det[:4]
            conf = det[4]
            cls_id = int(det[5])
            label = f"Mango {conf:.2f}" # class_names[cls_id] if multiple classes
            
            img_with_boxes = plot_one_box(xyxy, img_with_boxes, label=label, color=(0,255,0))
            
            # Store results including pixel width and height of the bbox
            w_px = (xyxy[2] - xyxy[0]).item()
            h_px = (xyxy[3] - xyxy[1]).item()
            detections_scaled.append(list(xyxy.cpu().numpy()) + [conf.item(), cls_id, w_px, h_px])


        if output_image_path:
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, img_with_boxes)
            print(f"Saved detection result to: {output_image_path}")
        
        return detections_scaled, img_with_boxes # Detections are [x1,y1,x2,y2,conf,cls, w_px,h_px]
    else:
        print(f"No detections for {image_path}")
        if output_image_path: # Save original image if no detections
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, img_orig)
        return [], img_orig


def main():
    # --- Configs ---
    with open("configs/model.yaml", 'r') as f:
        model_cfg = yaml.safe_load(f) # For UDIoU params if needed by NMS
    with open("configs/training.yaml", 'r') as f: # For device, checkpoint path
        training_cfg = yaml.safe_load(f)
    
    device = torch.device(training_cfg['device'] if torch.cuda.is_available() else "cpu")

    # --- Load Model ---
    from model.detection.detect_model import AEMangoYOLODetector # Import here
    model = AEMangoYOLODetector(cfg_model_yaml_path="configs/model.yaml", num_classes_override=1)
    
    checkpoint_path = os.path.join(training_cfg['detection_checkpoint_dir'], "ae_mangoyolo5_epoch_20.pt") # Example last epoch
    # Or find best checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}. Train the model first.")
        # Try to find any checkpoint
        checkpoints = sorted(glob.glob(os.path.join(training_cfg['detection_checkpoint_dir'], "*.pt")))
        if checkpoints:
            checkpoint_path = checkpoints[-1] # Take the latest one
            print(f"Using latest checkpoint: {checkpoint_path}")
        else:
            return

    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}, epoch {ckpt.get('epoch', 'N/A')}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}. Ensure model architecture matches checkpoint.")
        return
        
    model.to(device).eval()

    # --- Inference Parameters ---
    # These could also be in a separate inference_config.yaml
    nms_conf_threshold = model_cfg.get('nms_conf_threshold', 0.25) 
    nms_iou_threshold = model_cfg.get('nms_iou_threshold', 0.45) # For UDIoU comparison

    # --- Input Image / Directory ---
    # Replace with your image path or directory
    input_path = "path/to/your/test_image.jpg" # Or "path/to/your/test_image_directory"
    output_dir = "outputs/inference_results/detection"
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(input_path):
        import glob
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.extend(glob.glob(os.path.join(input_path, ext)))
        
        if not image_files:
            print(f"No images found in directory: {input_path}")
            return
        
        for img_file in image_files:
            print(f"\nProcessing {img_file}...")
            output_img_file = os.path.join(output_dir, os.path.basename(img_file))
            detections, _ = infer_single_image(img_file, model, device, model_cfg, nms_conf_threshold, nms_iou_threshold, output_img_file)
            if detections:
                print(f" Detections found: {len(detections)}")
                # for det in detections: print(det) # [x1,y1,x2,y2,conf,cls, w_px,h_px]
    elif os.path.isfile(input_path):
        output_img_file = os.path.join(output_dir, os.path.basename(input_path))
        detections, _ = infer_single_image(input_path, model, device, model_cfg, nms_conf_threshold, nms_iou_threshold, output_img_file)
        if detections:
            print(f"Detections for {input_path}: {len(detections)}")
            # for det in detections: print(det)
    else:
        print(f"Error: Input path {input_path} is not a valid file or directory.")

if __name__ == '__main__':
    # Create dummy configs and model checkpoint dir for a dry run
    # Ensure model.yaml, training.yaml exist in configs/
    # And a dummy checkpoint might be needed in outputs/checkpoints/detection/
    # For a real run, replace "path/to/your/test_image.jpg"
    
    # Example:
    # os.makedirs("configs", exist_ok=True)
    # if not os.path.exists("configs/model.yaml"):
    #     with open("configs/model.yaml", "w") as f: yaml.dump({'num_classes': 1, 'nms_conf_threshold':0.25, 'nms_iou_threshold':0.45}, f)
    # if not os.path.exists("configs/training.yaml"):
    #     with open("configs/training.yaml", "w") as f: yaml.dump({'device': 'cpu', 'detection_checkpoint_dir':'outputs/checkpoints/detection/'}, f)
    # os.makedirs("outputs/checkpoints/detection/", exist_ok=True)
    # # Create a dummy checkpoint if you want to run this without training first
    # # This requires a trained model structure.
    # # from model.detection.detect_model import AEMangoYOLODetector
    # # dummy_model = AEMangoYOLODetector(cfg_model_yaml_path="configs/model.yaml", num_classes_override=1)
    # # torch.save({'model_state_dict': dummy_model.state_dict(), 'epoch':0}, "outputs/checkpoints/detection/ae_mangoyolo5_epoch_0.pt")

    print("NOTE: Detection inference script. Ensure a trained model checkpoint exists.")
    print("Replace 'path/to/your/test_image.jpg' with an actual image path.")
    
    try:
        main()
    except FileNotFoundError as e:
        print(f"Missing file or directory: {e}. Please check paths and configs.")
    except Exception as e:
        print(f"An error occurred during detection inference: {e}")
        import traceback
        traceback.print_exc()
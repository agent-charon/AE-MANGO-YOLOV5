import torch
import torchvision
from .unified_diou import bbox_iou # Import from local utils

def non_max_suppression_udiou(predictions, conf_thres=0.25, iou_thres=0.45, udiou_params=None, multi_label=True, max_det=300):
    """
    Performs Non-Maximum Suppression (NMS) on inference results.
    The paper says "U-DIoU enhances non-max suppression", so we use UDIoU as the metric.
    Returns:
         list of detections, on (nDet, 6) tensor per image [xyxy, conf, cls]
    """
    # Input: (batch_size, num_anchors, 5+num_classes)
    # [cx, cy, w, h, obj_conf, class1_conf, class2_conf, ...]

    bs = predictions.shape[0]  # batch size
    nc = predictions.shape[2] - 5  # number of classes
    xc = predictions[..., 4] > conf_thres  # candidates by object confidence

    output = [torch.zeros((0, 6), device=predictions.device)] * bs

    for xi, x in enumerate(predictions):  # image index, image inference
        x = x[xc[xi]]  # filter by confidence

        if not x.shape[0]:
            continue

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        n = x.shape[0]  # number of boxes
        if not n:
            continue
            
        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)] # Should be already sorted if using torchvision.ops.nms

        # Batched NMS
        c = x[:, 5:6] * (4096)  #  Max classes for NMS trick (offsets for different classes)
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # NMS using UDIoU
        # torchvision.ops.nms needs IoU. We need to compute UDIoU matrix.
        # This is computationally more expensive than standard NMS.
        # Standard NMS (using simple IoU)
        # i = torchvision.ops.nms(boxes, scores, iou_thres) 
        
        # Custom NMS with UDIoU:
        # 1. Sort boxes by score
        # 2. Iteratively select highest score box, remove overlapping boxes using UDIoU
        keep_indices = []
        # Prepare boxes for UDIoU (needs to be xywh for our bbox_iou if it expects that)
        # Or adjust bbox_iou to handle xyxy. Our current bbox_iou converts xywh to xyxy.
        # For NMS, boxes are typically xyxy.
        
        # The `boxes` variable here are xyxy (offset by class `c`). We need original xyxy for UDIoU.
        original_boxes_xyxy = x[:, :4] 

        # Sort by score
        _, order = scores.sort(descending=True)
        
        while order.numel() > 0:
            if order.numel() == 1: # Last box
                idx = order.item()
                keep_indices.append(idx)
                break
            
            current_idx = order[0]
            keep_indices.append(current_idx)
            
            current_box = original_boxes_xyxy[current_idx].unsqueeze(0) # (1, 4)
            remaining_boxes = original_boxes_xyxy[order[1:]]       # (N-1, 4)
            
            if remaining_boxes.numel() == 0:
                break

            # Calculate UDIoU between current_box and remaining_boxes
            # Our bbox_iou needs xywh. Let's convert original_boxes_xyxy to xywh first for it.
            # Or better, modify bbox_iou to accept xyxy directly.
            # For now, let's assume original_boxes_xyxy are already in the format needed by a UDIoU calc
            # or pass xywh=False to bbox_iou.

            # UDIoU expects box1 (N,4) and box2 (M,4).
            # current_box (1,4), remaining_boxes (K,4)
            # udiou_values = ud_iou_metric(current_box, remaining_boxes) # This should return (1, K)
            # The ud_iou_metric should be a function that computes UDIoU specifically.
            # Our bbox_iou can compute UDIoU if UDIoU=True is passed.
            # It needs boxes in xywh by default, or xywh=False for xyxy.
            udiou_scores = bbox_iou(current_box, remaining_boxes, xywh=False, UDIoU=True, udiou_params=udiou_params) # Shape (1, K)
            
            # UDIoU is a similarity metric, higher is better.
            # For NMS, we discard if UDIoU > threshold.
            # Standard IoU threshold is used (iou_thres).
            # A higher UDIoU means more similar/overlapping.
            inds_to_keep = (udiou_scores.squeeze(0) <= iou_thres)
            order = order[1:][inds_to_keep] # Keep those with low UDIoU
            
        i = torch.tensor(keep_indices, device=predictions.device)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        
        output[xi] = x[i]
        
    return output

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

if __name__ == '__main__':
    # Example usage:
    # Mock predictions: (batch_size, num_detections_before_nms, 5 + num_classes)
    # 5 = [cx, cy, w, h, object_confidence]
    # num_classes = 1 (mango)
    bs = 1
    num_anchors_example = 100 
    num_classes = 1
    mock_preds = torch.rand(bs, num_anchors_example, 5 + num_classes)
    # Make some confidences higher
    mock_preds[0, :10, 4] = torch.rand(10) * 0.5 + 0.5 # Object confidence
    mock_preds[0, :10, 5] = torch.rand(10) * 0.5 + 0.5 # Class confidence
    
    # Make some boxes overlap
    mock_preds[0, 0, :4] = torch.tensor([0.5, 0.5, 0.2, 0.2]) # cx, cy, w, h
    mock_preds[0, 1, :4] = torch.tensor([0.51, 0.51, 0.2, 0.2])
    mock_preds[0, 2, :4] = torch.tensor([0.8, 0.8, 0.1, 0.1])


    udiou_params_example = { # From model.yaml (or sensible defaults)
        'v_aspect_term_coeff': 0.5,
        'lambda_scale_invariance_coeff': 0.3,
        'beta_p_center_dist_coeff': 0.2,
        'gamma_theta_orientation_coeff': 0.1
    }
    
    detections = non_max_suppression_udiou(mock_preds, conf_thres=0.4, iou_thres=0.5, udiou_params=udiou_params_example)
    for i, det in enumerate(detections):
        print(f"Image {i} Detections (xyxy, conf, cls):")
        if det.numel() > 0:
            print(det)
        else:
            print("No detections.")
import torch
import math
import numpy as np

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, UDIoU=False, eps=1e-7, udiou_params=None):
    """
    Calculate Intersection over Union (IoU) or its variants.
    Args:
        box1 (tensor): Predicted bounding boxes, shape (N, 4) or (B, N, 4)
        box2 (tensor): Ground truth bounding boxes, shape (M, 4) or (B, M, 4)
        xywh (bool): If true, boxes are in [x_center, y_center, width, height] format.
                     Otherwise, [x_min, y_min, x_max, y_max].
        GIoU (bool): Use Generalized IoU.
        DIoU (bool): Use Distance IoU.
        CIoU (bool): Use Complete IoU.
        UDIoU (bool): Use Unified DIoU (as per the paper).
        eps (float): Epsilon to prevent division by zero.
        udiou_params (dict): Parameters for UDIoU if UDIoU is True.
                             Expected keys: 'alpha_iou_weight', 'v_aspect_ratio_consistency',
                                            's_scale_invariance_lambda_tuning', 'beta_center_distance_penalty',
                                            'theta_angular_alignment_importance'.
                                            And weights for these components if used as a loss.
                                            For NMS, we just need the UDIoU value.
    Returns:
        (tensor): IoU values, shape (N, M) or (B, N, M)
    """
    # Convert boxes to (x_min, y_min, x_max, y_max)
    if xywh:
        # box1
        b1_x1, b1_y1 = box1[..., 0] - box1[..., 2] / 2, box1[..., 1] - box1[..., 3] / 2
        b1_x2, b1_y2 = box1[..., 0] + box1[..., 2] / 2, box1[..., 1] + box1[..., 3] / 2
        # box2
        b2_x1, b2_y1 = box2[..., 0] - box2[..., 2] / 2, box2[..., 1] - box2[..., 3] / 2
        b2_x2, b2_y2 = box2[..., 0] + box2[..., 2] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # Intersection area
    inter_x1 = torch.max(b1_x1.unsqueeze(-1), b2_x1.unsqueeze(-2))
    inter_y1 = torch.max(b1_y1.unsqueeze(-1), b2_y1.unsqueeze(-2))
    inter_x2 = torch.min(b1_x2.unsqueeze(-1), b2_x2.unsqueeze(-2))
    inter_y2 = torch.min(b1_y2.unsqueeze(-1), b2_y2.unsqueeze(-2))
    inter_w = torch.clamp(inter_x2 - inter_x1, min=0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0)
    intersection = inter_w * inter_h

    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area.unsqueeze(-1) + b2_area.unsqueeze(-2) - intersection + eps

    iou = intersection / union # Eq. 4 (modified)

    if not (GIoU or DIoU or CIoU or UDIoU):
        return iou

    # Enclosing box (smallest box covering both)
    c_x1 = torch.min(b1_x1.unsqueeze(-1), b2_x1.unsqueeze(-2))
    c_y1 = torch.min(b1_y1.unsqueeze(-1), b2_y1.unsqueeze(-2))
    c_x2 = torch.max(b1_x2.unsqueeze(-1), b2_x2.unsqueeze(-2))
    c_y2 = torch.max(b1_y2.unsqueeze(-1), b2_y2.unsqueeze(-2))
    c_w = c_x2 - c_x1
    c_h = c_y2 - c_y1

    if GIoU:
        c_area = c_w * c_h + eps
        return iou - (c_area - union) / c_area

    # Center points
    b1_cx, b1_cy = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
    b2_cx, b2_cy = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2

    # Diagonal length of enclosing box (c in Eq. 3)
    c_diag_sq = c_w**2 + c_h**2 + eps
    # Distance between center points (rho in Eq. 3)
    rho_sq = ((b1_cx.unsqueeze(-1) - b2_cx.unsqueeze(-2))**2 +
              (b1_cy.unsqueeze(-1) - b2_cy.unsqueeze(-2))**2)

    diou_term = rho_sq / c_diag_sq # This is p^2(b,b_gt)/c^2 from Eq. 3
    
    if DIoU:
        return iou - diou_term # Eq. 3 is DIoU = IoU - p^2/c^2

    # For CIoU and UDIoU
    b1_w, b1_h = b1_x2 - b1_x1, b1_y2 - b1_y1
    b2_w, b2_h = b2_x2 - b2_x1, b2_y2 - b2_y1

    # Aspect ratio term v for CIoU
    v = (4 / (math.pi**2)) * (torch.atan(b1_w.unsqueeze(-1) / (b1_h.unsqueeze(-1) + eps)) -
                               torch.atan(b2_w.unsqueeze(-2) / (b2_h.unsqueeze(-2) + eps)))**2
    alpha_ciou = v / (1 - iou + v + eps) # Weight for v
    
    if CIoU:
        return iou - diou_term - alpha_ciou * v
    
    if UDIoU:
        # Unified DIoU specific terms from Eq. 5 and Table 1
        # U-DIoU = IoU - p^2/c^2 - alpha*v - beta*Delta_s - gamma*Delta_theta
        # Paper: U-DIoU = IoU - p^2/c^2 - alpha*v - lambda*As - beta_p*(p(b,bgt)/c) - theta_diff
        # There seems to be a slight notation mismatch in text vs. Eq. 5. I'll follow Eq. 5.
        # U-DIoU = IoU - p^2(b,b_gt)/c^2 - alpha*v - lambda*Delta_s - beta_p*rho_p(b,b_gt)/c - gamma_theta*theta
        # where rho_p is the center distance.

        if udiou_params is None:
            # Default params if not provided, these need to be sensible defaults or from config
            udiou_params = {
                'v_aspect_term_coeff': 1.0, # alpha in Eq. 5, coefficient for v
                'v_aspect_ratio_consistency_weight': 4 / (math.pi**2), # v calculation part
                'lambda_scale_invariance_coeff': 1.0, # lambda in Eq. 5
                # As term (scale invariance) log(Area(b)/Area(b_gt)) - paper says Area(b)/Area(b_gt)
                # Table 1 says log(Area(b)/Area(b_gt)). Let's use log.
                'beta_p_center_dist_coeff': 1.0, # beta_p in Eq. 5 (linear penalty based on center distance)
                'gamma_theta_orientation_coeff': 1.0 # gamma_theta in Eq. 5
            }
        
        # v_aspect_ratio_term (alpha*v from Eq.5, Table 1: v = (4/pi^2)(arctan(w_gt/h_gt) - arctan(w/h))^2 )
        # Note: CIoU's v is (arctan(w1/h1) - arctan(w2/h2))^2. Paper says (arctan(w_gt/h_gt) - arctan(w/h))^2
        # This matches v from CIoU if we map gt to box2 and pred to box1.
        # alpha in Eq. 5 is a weight for v. Table 1 gives alpha as "Weight balancing aspect ratio influence based on IoU"
        # alpha_udiou = (1 - iou) / (1 - iou + v + eps) # Example, similar to CIoU's alpha
        # For simplicity, let's use udiou_params['v_aspect_term_coeff'] as the direct alpha for v
        v_term = udiou_params.get('v_aspect_term_coeff', 1.0) * v

        # As term (scale invariance from Eq.5, Table 1: As = log(Area(b)/Area(b_gt)))
        # Lambda is a tuning parameter for scale invariance.
        area_b1 = b1_w * b1_h
        area_b2 = b2_w * b2_h
        # As = torch.log(area_b1.unsqueeze(-1) / (area_b2.unsqueeze(-2) + eps) + eps) # As in Eq. 5
        As = torch.abs(torch.log(area_b1.unsqueeze(-1) / (area_b2.unsqueeze(-2) + eps) + eps)) # Ensure positive penalty
        lambda_As_term = udiou_params.get('lambda_scale_invariance_coeff', 1.0) * As
        
        # rho_p(b,bgt)/c term (linear penalty for center distance, Eq.5)
        # rho_p is sqrt(rho_sq). c_diag is sqrt(c_diag_sq)
        rho_p_div_c_term = udiou_params.get('beta_p_center_dist_coeff', 1.0) * (torch.sqrt(rho_sq) / (torch.sqrt(c_diag_sq) + eps))
        
        # theta term (angular difference, Eq.5)
        # This is complex if boxes are not axis-aligned. For axis-aligned, theta diff is 0 or pi/2.
        # The paper doesn't specify how theta is calculated for axis-aligned bounding boxes.
        # For axis-aligned, orientation difference is usually not considered or is implicitly handled by aspect ratio.
        # If we assume orientation means portrait vs landscape:
        # orientation_b1 = (b1_h > b1_w).float() # 1 if portrait, 0 if landscape
        # orientation_b2 = (b2_h > b2_w).float()
        # theta_diff = torch.abs(orientation_b1.unsqueeze(-1) - orientation_b2.unsqueeze(-2)) * (math.pi / 2) # 0 or pi/2
        # This is a simplification. If actual rotated boxes are used, then proper angle calculation is needed.
        # For simplicity, let's assume theta term is small or handled by 'v' for axis-aligned boxes.
        # Or, if theta is angle of diagonal:
        theta_b1 = torch.atan2(b1_h, b1_w)
        theta_b2 = torch.atan2(b2_h.unsqueeze(-2), b2_w.unsqueeze(-2)) # Ensure broadcasting
        theta_diff_term = udiou_params.get('gamma_theta_orientation_coeff', 1.0) * torch.abs(theta_b1.unsqueeze(-1) - theta_b2)

        # UDIoU from Eq. 5: IoU - (rho_sq/c_diag_sq) - v_term - lambda_As_term - rho_p_div_c_term - theta_diff_term
        # Note the signs are penalties.
        udiou = iou - diou_term - v_term - lambda_As_term - rho_p_div_c_term - theta_diff_term
        return udiou

    return iou # Should not reach here if a variant was selected

if __name__ == '__main__':
    # Example usage:
    # Predicted boxes (batch of 1, 2 boxes)
    pred_boxes_xywh = torch.tensor([[[100, 100, 50, 50], [200, 200, 60, 40]]], dtype=torch.float32)
    # Ground truth boxes (batch of 1, 3 boxes)
    gt_boxes_xywh = torch.tensor([[[110, 110, 50, 50], [210, 210, 60, 40], [300, 300, 30, 30]]], dtype=torch.float32)

    iou_val = bbox_iou(pred_boxes_xywh, gt_boxes_xywh, xywh=True)
    print("IoU:\n", iou_val)

    diou_val = bbox_iou(pred_boxes_xywh, gt_boxes_xywh, xywh=True, DIoU=True)
    print("DIoU:\n", diou_val) # DIoU = IoU - penalty

    ciou_val = bbox_iou(pred_boxes_xywh, gt_boxes_xywh, xywh=True, CIoU=True)
    print("CIoU:\n", ciou_val) # CIoU = IoU - penalty1 - penalty2

    # UDIoU example (params need to be sensible)
    udiou_params_example = {
        'v_aspect_term_coeff': 0.5,
        'lambda_scale_invariance_coeff': 0.3,
        'beta_p_center_dist_coeff': 0.2,
        'gamma_theta_orientation_coeff': 0.1
    }
    udiou_val = bbox_iou(pred_boxes_xywh, gt_boxes_xywh, xywh=True, UDIoU=True, udiou_params=udiou_params_example)
    print("UDIoU:\n", udiou_val)

    # For DIoU term in Lpos (predicting single mango)
    # pred_box (1,4), gt_box (1,4)
    pred_b = torch.tensor([[100., 100., 50., 50.]])
    gt_b   = torch.tensor([[110., 110., 45., 45.]])
    
    iou_single = bbox_iou(pred_b, gt_b, xywh=True)
    diou_single = bbox_iou(pred_b, gt_b, xywh=True, DIoU=True)
    # The DIoU term for Lpos is p^2/c^2, so it's IoU - DIoU_value
    diou_penalty_term = iou_single - diou_single
    print(f"IoU (single): {iou_single.item()}")
    print(f"DIoU value (single): {diou_single.item()}")
    print(f"DIoU penalty term p^2/c^2 (for Lpos): {diou_penalty_term.item()}")
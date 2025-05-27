import torch
import torch.nn as nn
from model.detection.backbone.modified_yolov5s_backbone import AEMangoYOLO5Backbone
# from model.detection.utils.nms import non_max_suppression_udiou # For inference
# For training, loss computation is key.

class Detect(nn.Module):
    # Standard YOLOv5 Detect layer
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=()):  # number_classes, anchors, channels
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor (cx, cy, w, h, conf, cls0...clsN)
        self.nl = len(anchors)  # number of detection layers (scales)
        self.na = len(anchors[0]) // 2  # number of anchors per scale
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        
        # Store anchors as Parameter to be part of the model state
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

    def forward(self, x):
        # x is a list of feature maps from the neck (P2, P3, P4, P5)
        z = []  # inference output
        for i in range(self.nl): # For each detection scale
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # batch_size, C, H, W
            # Reshape to (bs, num_anchors, H, W, num_outputs)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no)) # Flatten detections for this scale

        return x if self.training else (torch.cat(z, 1), x) # Train: raw outputs, Infer: processed outputs + raw

    def _make_grid(self, nx=20, ny=20, i=0):
        # device = self.anchors[i].device
        # Correct way to get device if anchors buffer is used:
        device = self.anchors.device
        
        yv, xv = torch.meshgrid([torch.arange(ny, device=device), torch.arange(nx, device=device)], indexing='ij')
        grid = torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]) \
            .view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class AEMangoYOLODetector(nn.Module):
    def __init__(self, cfg_model_yaml_path, num_classes_override=None):
        super().__init__()
        # For simplicity, hardcoding some parameters here.
        # In a real scenario, load these from model.yaml
        # Example anchors for 4 scales (P2, P3, P4, P5)
        # These need to be tuned based on your dataset (mango sizes at different scales)
        self.anchors_example = [ 
            [5,6, 8,10, 10,14],         # P2/4 (for 160x160 head, small anchors)
            [10,13, 16,30, 33,23],      # P3/8
            [30,61, 62,45, 59,119],     # P4/16
            [116,90, 156,198, 373,326]  # P5/32
        ]
        # Channel sizes of feature maps from backbone neck output
        # [P2_ch, P3_ch, P4_ch, P5_ch]
        self.ch_example = [64, 128, 256, 512] # Must match AEMangoYOLO5Backbone output channels

        # Strides for each feature map
        self.strides_example = torch.tensor([4., 8., 16., 32.]) # Corresponding to P2, P3, P4, P5

        nc = num_classes_override if num_classes_override is not None else 1 # Mango class

        self.backbone = AEMangoYOLO5Backbone(num_classes=nc, variant_is_mango_yolo5=True) # As per paper
        
        # The Detect layer needs to know the strides for anchor scaling.
        # Usually, this is passed from the main model class that holds the backbone and detect head.
        self.detect_head = Detect(nc=nc, anchors=self.anchors_example, ch=self.ch_example)
        self.detect_head.stride = self.strides_example # Set strides in Detect layer

        # TODO: Initialize weights (see YOLOv5 official repo for common init schemes)
        # self.apply(self._initialize_weights)


    def forward(self, x, targets=None): # targets are for training loss
        features = self.backbone(x)
        predictions = self.detect_head(features) # List of raw outputs if training, (processed, raw) if eval
        return predictions

    # def _initialize_weights(self, m):
    #     # Placeholder for weight initialization
    #     if isinstance(m, nn.Conv2d):
    #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         if m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.BatchNorm2d):
    #         nn.init.constant_(m.weight, 1)
    #         nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight, 0, 0.01)
    #         nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # Example:
    # Create a dummy config path for model.yaml if needed for AEMangoYOLO5Backbone
    # For this example, AEMangoYOLO5Backbone is self-contained.
    
    model = AEMangoYOLODetector(cfg_model_yaml_path=None, num_classes_override=1)
    model.eval() # Set to evaluation mode for inference output format

    dummy_input_image = torch.randn(1, 3, 640, 640) # B, C, H, W
    
    with torch.no_grad():
        # predictions_infer: (tensor of detections, list of raw feature maps)
        predictions_infer, raw_features_infer = model(dummy_input_image) 
    
    print("Inference mode:")
    print("Processed Detections Shape (Batch, Num_Detections_Total, Num_Outputs_per_Anchor):", predictions_infer.shape)
    # print("Raw Feature Maps from Detect Head (list, one per scale):")
    # for i, feat in enumerate(raw_features_infer):
    #     print(f" Scale {i} raw output shape: {feat.shape}")

    model.train() # Set to training mode
    # predictions_train: list of raw feature maps from Detect head
    predictions_train = model(dummy_input_image)
    print("\nTraining mode:")
    print("Raw Feature Maps from Detect Head (list, one per scale):")
    for i, feat in enumerate(predictions_train):
        print(f" Scale {i} raw output shape: {feat.shape}")
    
    # To use NMS:
    # from model.detection.utils.nms import non_max_suppression_udiou
    # udiou_params_example = {
    #     'v_aspect_term_coeff': 0.5, 'lambda_scale_invariance_coeff': 0.3,
    #     'beta_p_center_dist_coeff': 0.2, 'gamma_theta_orientation_coeff': 0.1
    # }
    # if not model.training:
    #     # NMS is applied to processed_detections
    #     # Processed detections are [xywh, conf, cls_scores...]
    #     # non_max_suppression_udiou expects (batch_size, num_anchors, 5+num_classes)
    #     # The output of y.view(bs, -1, self.no) from Detect layer is suitable.
    #     final_detections = non_max_suppression_udiou(predictions_infer, conf_thres=0.25, iou_thres=0.45, udiou_params=udiou_params_example)
    #     print("\nNMS Output (example):", final_detections[0].shape if final_detections[0].numel() > 0 else "No detections after NMS")
import torch
import torch.nn as nn
from model.detection.attention.cam import CAM
from model.detection.attention.sam import SAM

# --- Helper Modules (Standard YOLOv5 components - simplified for brevity) ---
def autopad(k, p=None):  # kernel, padding
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    #Paper: "removing 11 convolutional layers from the BottleneckCSP module"
    # This means the number of Bottleneck repetitions (n) or internal structure is reduced.
    # For this example, I'll assume n is reduced.
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, reduced_convs=False):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        
        # Number of bottlenecks 'n' is typically 3 for C3 in YOLOv5s.
        # If 11 convs are removed, 'n' would be significantly smaller, or internal structure changed.
        # A Bottleneck has 2 Conv layers. C3 has 'n' Bottlenecks.
        # So, if original n=3 (6 convs in bottlenecks), plus ~4 other convs in C3 structure = ~10 convs.
        # "Removing 11 convs" is a very large reduction.
        # For simplicity, let's assume n=1 if reduced_convs is True.
        actual_n = 1 if reduced_convs else n # This is a guess based on "11 convs removed"
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(actual_n)))


    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class Focus(nn.Module):
    # Focus wh information into c-space
    # Paper: "two layers from the focus module"
    # Original Focus: Conv(c1 * 4, c2, k=3) -> Conv(12, 32, 3) for YOLOv5s (1 layer)
    # "two layers removed" might mean it's simplified or replaced.
    # For this, let's assume a simplified version if reduced_convs.
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, reduced_convs=False):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        if reduced_convs: # Simplified Focus
             self.conv = Conv(c1 * 4, c2, k=1, act=act) # Reduced kernel size to 1 from 3
        else:
            self.conv = Conv(c1 * 4, c2, k=3, act=act) # Original Focus has k=3


    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class SPP(nn.Module):
    # Spatial Pyramid Pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        return torch.cat(x, self.d)

# --- AE-MangoYOLO5 Backbone and Neck ---
# This needs to be adapted from a full YOLOv5 model structure.
# Figure 2 in the paper is the guide.
class AEMangoYOLO5Backbone(nn.Module):
    def __init__(self, num_classes, ch=3, r_ratio=16, sam_k=7, variant_is_mango_yolo5=True):
        super().__init__()
        # These are example channel numbers, matching roughly YOLOv5s
        # variant_is_mango_yolo5 controls if Focus and BottleneckCSP are reduced
        
        # Backbone
        self.focus = Focus(ch, 32, k=3, reduced_convs=variant_is_mango_yolo5) # P1/2
        self.conv1 = Conv(32, 64, k=3, s=2) # P2/4
        self.cam1 = CAM(64, reduction_ratio=r_ratio)
        self.bottleneck_csp1 = BottleneckCSP(64, 64, n=1 if variant_is_mango_yolo5 else 3, reduced_convs=variant_is_mango_yolo5) # P2/4
        
        self.conv2 = Conv(64, 128, k=3, s=2) # P3/8
        self.cam2 = CAM(128, reduction_ratio=r_ratio)
        self.bottleneck_csp2 = BottleneckCSP(128, 128, n=2 if variant_is_mango_yolo5 else 6, reduced_convs=variant_is_mango_yolo5) # P3/8
        
        self.conv3 = Conv(128, 256, k=3, s=2) # P4/16
        self.bottleneck_csp3 = BottleneckCSP(256, 256, n=2 if variant_is_mango_yolo5 else 6, reduced_convs=variant_is_mango_yolo5) # P4/16
        
        self.conv4 = Conv(256, 512, k=3, s=2) # P5/32
        self.spp = SPP(512, 512)
        self.bottleneck_csp4 = BottleneckCSP(512, 512, n=1 if variant_is_mango_yolo5 else 3, reduced_convs=variant_is_mango_yolo5) # P5/32

        # Neck (Path Aggregation Network - PANet structure)
        # Upsample + Concat + CSP + Conv
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat1 = Concat(dimension=1) # To concat upsampled features with backbone features
        self.sam1 = SAM(kernel_size=sam_k)
        self.neck_csp1 = BottleneckCSP(512 + 256, 256, n=1, shortcut=False, reduced_convs=variant_is_mango_yolo5) # Input channels depend on backbone output P4

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.concat2 = Concat(dimension=1)
        self.sam2 = SAM(kernel_size=sam_k)
        self.neck_csp2 = BottleneckCSP(256 + 128, 128, n=1, shortcut=False, reduced_convs=variant_is_mango_yolo5) # Output for small object detection head (P3 feature map)

        # Downsample path
        self.neck_conv1 = Conv(128, 128, k=3, s=2)
        self.concat3 = Concat(dimension=1)
        self.sam3 = SAM(kernel_size=sam_k)
        self.neck_csp3 = BottleneckCSP(128 + 256, 256, n=1, shortcut=False, reduced_convs=variant_is_mango_yolo5) # Output for medium object detection head (P4 feature map)

        self.neck_conv2 = Conv(256, 256, k=3, s=2)
        self.concat4 = Concat(dimension=1)
        # No SAM after last downsample according to Fig. 2 before SPP block in original YOLOv5 like structure
        self.neck_csp4 = BottleneckCSP(256 + 512, 512, n=1, shortcut=False, reduced_convs=variant_is_mango_yolo5) # Output for large object detection head (P5 feature map)
        
        # Additional 160x160 detection layer head
        # This implies using an earlier feature map, e.g., from self.conv1 (P2/4) or self.bottleneck_csp1
        # Let's assume it taps off self.bottleneck_csp1 (64 channels, /4 stride)
        # We need to process this to be suitable for a detection head.
        # For example, upsample P3 features or downsample P2 features.
        # Fig 2 shows a "160x160" detection head. YOLOv5 default heads are for 80x80, 40x40, 20x20.
        # If input is 640x640:
        # P2 (/4) -> 160x160. Features from bottleneck_csp1.
        # P3 (/8) -> 80x80. Features from neck_csp2.
        # P4 (/16) -> 40x40. Features from neck_csp3.
        # P5 (/32) -> 20x20. Features from neck_csp4.

        # So, the 160x160 head would be on features from P2 (e.g., self.bottleneck_csp1)
        # Let's define a small conv sequence for this new head.
        # The figure doesn't clearly show how this 160x160 head is integrated into the neck FPN/PAN.
        # Let's assume it taps off the bottleneck_csp1 and has its own small path.
        self.head_160_conv = Conv(64, 64, k=1, s=1) # Example, adjust channels as needed for Detect layer

        # Detection Heads (implemented in DetectModel)
        # self.detect = Detect(...)
        
        # Initialize weights (common practice)
        # self._initialize_weights()


    def forward(self, x):
        # Backbone
        x_focus = self.focus(x)
        x_c1 = self.conv1(x_focus)
        x_csp1_out = self.bottleneck_csp1(self.cam1(x_c1)) # P2 feature for 160x160 head
        
        x_c2 = self.conv2(x_csp1_out)
        x_csp2_out = self.bottleneck_csp2(self.cam2(x_c2)) # P3 feature
        
        x_c3 = self.conv3(x_csp2_out)
        x_csp3_out = self.bottleneck_csp3(x_c3) # P4 feature
        
        x_c4 = self.conv4(x_csp3_out)
        x_spp_out = self.spp(x_c4)
        x_csp4_out = self.bottleneck_csp4(x_spp_out) # P5 feature

        # Neck
        # Upsample path
        up_p5 = self.up1(x_csp4_out) # from P5 to P4 size
        cat_p4 = self.concat1([up_p5, x_csp3_out]) 
        neck_csp1_out = self.neck_csp1(self.sam1(cat_p4))

        up_p4 = self.up2(neck_csp1_out) # from P4 to P3 size
        cat_p3 = self.concat2([up_p4, x_csp2_out])
        p3_features = self.neck_csp2(self.sam2(cat_p3)) # Output for 80x80 detection head (stride 8)

        # Downsample path
        down_p3 = self.neck_conv1(p3_features)
        cat_p4_down = self.concat3([down_p3, neck_csp1_out]) # x_csp3_out was used in upsample, here neck_csp1_out
        p4_features = self.neck_csp3(self.sam3(cat_p4_down)) # Output for 40x40 detection head (stride 16)

        down_p4 = self.neck_conv2(p4_features)
        cat_p5_down = self.concat4([down_p4, x_csp4_out]) # x_csp4_out used here
        p5_features = self.neck_csp4(cat_p5_down) # Output for 20x20 detection head (stride 32)
        
        # Features for the 160x160 head from P2 (stride 4)
        p2_features_for_head = self.head_160_conv(x_csp1_out)

        return [p2_features_for_head, p3_features, p4_features, p5_features] # Strides /4, /8, /16, /32

    # def _initialize_weights(self):
    #     # TODO: Implement weight initialization if needed
    #     pass

if __name__ == '__main__':
    # Example usage:
    # This is a conceptual backbone. A full Detect class is needed to make it runnable.
    # It depends on how Detect head is implemented to consume these features.
    model = AEMangoYOLO5Backbone(num_classes=1, variant_is_mango_yolo5=True)
    # print(model)
    dummy_input = torch.randn(1, 3, 640, 640) # Example input size (B,C,H,W)
    
    # To make this runnable, we need a Detect layer that matches these output feature channels.
    # For now, let's just print the shapes of the output feature maps.
    # This requires defining the output channels of Conv/CSP layers correctly.
    # The channel numbers (32, 64, 128, 256, 512) are typical for YOLOv5s.
    # The output channels of neck_csp1, neck_csp2 etc. should match what Detect layer expects.
    
    # Let's assume the output channels of neck_csp are:
    # neck_csp2 (P3 features for 80x80 head) -> 128 channels
    # neck_csp3 (P4 features for 40x40 head) -> 256 channels
    # neck_csp4 (P5 features for 20x20 head) -> 512 channels
    # head_160_conv (P2 features for 160x160 head) -> 64 channels
    
    try:
        outputs = model(dummy_input)
        print("Output feature map shapes (P2, P3, P4, P5 like):")
        for i, out_feat in enumerate(outputs):
            print(f"Layer {i+1}: {out_feat.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
        print("This backbone is conceptual and needs to be integrated with a Detect head.")
        print("The channel dimensions and connections need careful verification against a full YOLOv5 model.")
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x # Multiply attention map with input

class CAM(nn.Module):
    """
    Channel Attention Module (CAM) as described in Figure 3(a) and Eq. (1)
    The paper's Eq (1) is Mc(F) = sigma(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    And it's applied as F' = Mc(F) * F.
    The implementation detail of fc being Conv2d with kernel 1 is common.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAM, self).__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        self.hidden_channels = in_channels // reduction_ratio

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, self.hidden_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_channels, in_channels, bias=False)
        )
        # For Conv2d version of MLP often used with feature maps
        # self.mlp_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, self.hidden_channels, kernel_size=1, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(self.hidden_channels, in_channels, kernel_size=1, bias=False)
        # )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling
        avg_pool = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1) # B x C
        # Global Max Pooling
        max_pool = F.adaptive_max_pool2d(x, (1, 1)).view(x.size(0), -1) # B x C
        
        # Pass through MLP
        avg_mlp_out = self.mlp(avg_pool)
        max_mlp_out = self.mlp(max_pool)

        # Sum and apply sigmoid
        channel_attention_map = self.sigmoid(avg_mlp_out + max_mlp_out) # B x C
        
        # Reshape map and apply
        channel_attention_map = channel_attention_map.unsqueeze(2).unsqueeze(3) # B x C x 1 x 1
        return x * channel_attention_map.expand_as(x)

if __name__ == '__main__':
    # Example usage
    dummy_input = torch.randn(4, 64, 32, 32) # B, C, H, W
    cam_module = CAM(in_channels=64, reduction_ratio=16)
    output = cam_module(dummy_input)
    print("CAM Input shape:", dummy_input.shape)
    print("CAM Output shape:", output.shape) # Should be same as input
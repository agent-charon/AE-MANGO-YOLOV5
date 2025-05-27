import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        return self.sigmoid(y) * x # Multiply attention map with input

class SAM(nn.Module):
    """
    Spatial Attention Module (SAM) as described in Figure 3(b) and Eq. (2)
    Ms(F) = sigma(f_conv_7x7([AvgPool(F); MaxPool(F)]))
    Applied as F'' = Ms(F) * F'
    """
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        padding = kernel_size // 2
        
        # Convolutional layer to reduce channel from 2 to 1
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, 
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average pooling across channels
        avg_pool = torch.mean(x, dim=1, keepdim=True) # B x 1 x H x W
        # Max pooling across channels
        max_pool, _ = torch.max(x, dim=1, keepdim=True) # B x 1 x H x W
        
        # Concatenate
        concat_features = torch.cat([avg_pool, max_pool], dim=1) # B x 2 x H x W
        
        # Convolution and sigmoid
        spatial_attention_map = self.sigmoid(self.conv(concat_features)) # B x 1 x H x W
        
        # Apply attention
        return x * spatial_attention_map.expand_as(x)

if __name__ == '__main__':
    # Example usage
    dummy_input = torch.randn(4, 64, 32, 32) # B, C, H, W
    sam_module = SAM(kernel_size=7)
    output = sam_module(dummy_input)
    print("SAM Input shape:", dummy_input.shape)
    print("SAM Output shape:", output.shape) # Should be same as input
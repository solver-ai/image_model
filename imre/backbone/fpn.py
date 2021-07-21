import torch
import torch.nn as nn
import torchvision
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

class FPN(nn.Module):
    def __init__(
        self, in_channels_list, out_channels
    ):
        super(FPN, self).__init__()
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list,
            out_channels,
            LastLevelP6P7(out_channels, out_channels)
        )  
            
    def forward(self, x:dict) -> list:
        features = self.fpn(x)
        return list(features.values())
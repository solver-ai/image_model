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

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)

            
    def forward(self, x:dict) -> list:
        x = {f'layer{idx}':i for idx, i in enumerate(x,2)}
        features = self.fpn(x)
        return list(features.values())
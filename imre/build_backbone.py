# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from imre.backbone.fpn import FPN
from imre.backbone import resnet

def build_resnet_fpn_p3p7_backbone(cfg):
    # body = resnet.ResNet(cfg)
    body = getattr(resnet, cfg.MODEL.BACKBONE.MODEL_NAME)()
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 
    fpn = FPN(
        in_channels_list=[
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    return build_resnet_fpn_p3p7_backbone(cfg)


if __name__ == "__main__":
    from config import _C
    from copy import deepcopy
    import torch
    cfg = deepcopy(_C)
    cfg.merge_from_file("atss/configs/atss_dcnv2_R_50_FPN_1x.yaml")
    input = torch.rand([4,3,256,256], dtype=torch.float32)
    
    backbone = build_backbone(cfg)
    output = backbone(input)
    print(backbone.out_channels)
    print([layers.shape for layers in output])
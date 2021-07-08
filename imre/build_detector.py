import torch
from torch import nn

from atss_core.structures.image_list import to_image_list

import build_backbone
from atss.build_atss import ATSSModule

class ATSS(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.backbone = build_backbone(cfg)
        self.detector = ATSSModule(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        proposals, proposal_losses = self.detector(images, features, targets)

        x = features
        result = proposals
        detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
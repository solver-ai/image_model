import torch
from torch import nn

from build_backbone import build_backbone
from module.utils import to_image_list
from atss.build_atss import ATSSModule

class ATSS(nn.Module):
    def __init__(self, cfg):
        super(ATSS, self).__init__()

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

if __name__ == "__main__":
    
    from atss.configs.default import _C
    from copy import deepcopy
    import torch
    cfg = deepcopy(_C)
    cfg.merge_from_file("atss/configs/atss_R_50_FPN_1x.yaml")
    input = torch.rand([4,3,256,256], dtype=torch.float32)
    
    detector = ATSS(cfg)
    detector.eval()
    output = detector(input)
    print(output)
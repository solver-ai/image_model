import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF

from imre.atss.build_atss import ATSSModule
from imre.build_backbone import build_backbone 
from imre.module.utils import to_image_list

class ATSSModel(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super(ATSSModel, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.rpn = ATSSModule(cfg, self.backbone.out_channels)
    
    def forward(self, x, y):
        x = to_image_list(x)
        features = self.backbone(x.tensors)
        proposals, proposal_losses = self.rpn(x, features, y)
        return proposals, proposal_losses

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch

        _, loss_dict = self(images, targets)
        loss_dict['loss'] = sum(loss for loss in loss_dict.values())
        print(loss_dict)
        
        return loss_dict
    
    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        boxes, _ = self(images, targets)
        return boxes

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, momentum=self.cfg.SOLVER.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1**(epoch // 30))
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF

from imre.atss.build_atss import BoxCoder, ATSSHead
from imre.backbone.build_backbone import build_backbone 

from imre.atss.loss import make_atss_loss_evaluator
from imre.atss.inference import make_atss_postprocessor
from imre.atss.anchor_generator import make_anchor_generator_atss

class ATSSModel(pl.LightningModule):
    def __init__(self, cfg, *args, **kwargs):
        super(ATSSModel, self).__init__()
        self.cfg = cfg

        #### build backbone ####
        self.backbone = build_backbone(cfg)

        #### build detector ####
        self.head = ATSSHead(cfg, self.backbone.out_channels)
        
        #### build loss ####
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder)
        self.anchor_generator = make_anchor_generator_atss(cfg)
    
    def forward(self, x, y):
        features = self.backbone(x)
        box_cls, box_regression, box_centerness = self.head(features)
        anchors = self.anchor_generator(x, features)
        return box_cls, box_regression, box_centerness, anchors

    def training_step(self, batch, batch_idx):
        images, targets, _ = batch
        box_cls, box_regression, box_centerness, anchors = self(images, targets)
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            box_cls, box_regression, box_centerness, targets, anchors
        )
        
        losses = {
            "loss": loss_box_cls + loss_box_reg + loss_centerness,
            "loss_reg": loss_box_reg,
            "loss_cls": loss_box_cls,
            "loss_centerness": loss_centerness,
        }
        print(losses)
        return losses
    
    def validation_step(self, batch, batch_idx):
        images, targets, _ = batch
        box_cls, box_regression, box_centerness, anchors = self(images, targets)
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            box_cls, box_regression, box_centerness, targets, anchors
        )

        if targets:
            loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
                box_cls, box_regression, box_centerness, targets, anchors
            )
            
            losses = {
                "loss": loss_box_cls + loss_box_reg + loss_centerness,
                "loss_reg": loss_box_reg,
                "loss_cls": loss_box_cls,
                "loss_centerness": loss_centerness,
            }
        
        boxes = self.box_selector_test(
            box_cls, box_regression, box_centerness,anchors
        )
        print(losses)
        return boxes, losses

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, momentum=self.cfg.SOLVER.MOMENTUM)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10,20], gamma=.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

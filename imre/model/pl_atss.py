import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF

from imre.atss.build_atss import ATSSHead, BoxCoder
from imre.build_backbone import build_backbone 
from imre.atss.loss import ATSSLossComputation
from imre.atss.inference import make_atss_postprocessor
from imre.atss.anchor_generator import make_anchor_generator_atss

class ATSSModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(ATSSModel, self).__init__()
        
        self.cfg = config
        self.backbone = build_backbone(self.cfg)
        self.head = ATSSHead(self.cfg, self.backbone.out_channels)

        box_coder = BoxCoder(self.cfg)
        self.loss_evaluator = ATSSLossComputation(self.cfg, box_coder)
        self.box_selector_test = make_atss_postprocessor(self.cfg, box_coder)
        self.anchor_generator = make_anchor_generator_atss(self.cfg)
        
    def forward(self, x):
        features = self.backbone(x)
        box_cls, box_regression, centerness = self.head(features)
        anchors = self.anchor_generator(x, features)
        return box_cls, box_regression, centerness, anchors

    def training_step(self, batch, batch_idx):
        images, targets = batch
        #### targets type check ####

        box_cls, box_regression, centerness, anchors  = self(torch.stack(images))
        
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors
        )

        self.log('train_loss_cls', loss_box_cls)
        self.log('train_loss_reg', loss_box_reg)
        self.log('train_loss_centerness', loss_centerness)
        losses = {
            'loss_cls' : loss_box_cls,
            'loss_reg' : loss_box_reg,
            'loss_centerness' : loss_centerness
        }
        return losses
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        box_cls, box_regression, centerness, anchors  = self(torch.stack(images))

        boxes = self.box_selector_test(
            box_cls, box_regression, centerness, anchors
        )

        return boxes

    #### check (mAP 실행?)
    # def validation_epoch_end(self, outputs):
    #     avg_loss = torch.stack([o["loss"] for o in outputs]).mean(-1)
    #     avg_acc = torch.stack([o["acc"] for o in outputs]).mean(-1)
    #     self.log("avg_val_loss", avg_loss)
    #     self.log("avg_val_acc", avg_acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.cfg.SOLVER.BASE_LR, momentum=.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1**(epoch // 30))
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
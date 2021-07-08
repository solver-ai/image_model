from imre.module import resnet

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import pytorch_lightning.metrics.functional as plF


class ResnetModel(pl.LightningModule):
    def __init__(self, config, *args, **kwargs):
        super(ResnetModel, self).__init__()
        
        self.config = config
        self.backbone = getattr(resnet, config.model)()
        self.loss = torch.nn.CrossEntropyLoss()
        
    def forward(self, x):
        out = self.backbone(x)
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        output,_ = self(x)
        acc = plF.accuracy(output, y)
        loss = self.loss(output, y)
        
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        # if self.lr_scheduler:
        #     self.log('learning_rate', self.lr_scheduler["scheduler"].get_last_lr()[-1])
        return {'loss':loss, 'acc':acc}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output,_ = self(x)
        acc = plF.accuracy(output, y)
        loss = self.loss(output, y)
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        return {'loss':loss, 'acc':acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([o["loss"] for o in outputs]).mean(-1)
        avg_acc = torch.stack([o["acc"] for o in outputs]).mean(-1)
        self.log("avg_val_loss", avg_loss)
        self.log("avg_val_acc", avg_acc)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.config.base_lr, momentum=.9)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1**(epoch // 30))
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
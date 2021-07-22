
import argparse

from imre.atss.configs.default import _C as cfg
from imre.dataset.detection_datasets import COCODataset
from imre.model.pl_atss import ATSSModel
from imre.module.utils import BatchCollator

import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


IMG_TRANSFORMS_PIPELINE = {
    "train" : A.Compose([
        A.LongestMaxSize(512),
        A.PadIfNeeded(
            min_height=640,
            min_width=640, 
            position='top_left', 
            border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()], bbox_params=A.BboxParams(format='coco')
    ),
    
    "valid" : A.Compose([
        A.LongestMaxSize(512),
        A.PadIfNeeded(
            min_height=640,
            min_width=640, 
            position='top_left', 
            border_mode=cv2.BORDER_CONSTANT),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        ToTensorV2()], bbox_params=A.BboxParams(format='coco')
    )
}


def get_dataset(data_path = "datasets"):
    train_dataset = COCODataset(
        ann_file='../datasets/deepfashion2/train.json',
        root='../datasets/deepfashion2/train',
        transforms=IMG_TRANSFORMS_PIPELINE['train']
    )
    valid_dataset = COCODataset(
        ann_file='../datasets/deepfashion2/valid.json', 
        root='../datasets/deepfashion2/valid',
        transforms=IMG_TRANSFORMS_PIPELINE['valid']
    )
    return train_dataset, valid_dataset


def train(cfg):

    train_dataset, valid_dataset = get_dataset() #data_path=cfg.DATA_PATH)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH


    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=BatchCollator,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=BatchCollator,
    )

    # fearture learning preparation
    model = ATSSModel(cfg)
    tb_logger = pl_loggers.TensorBoardLogger(
        "artifacts/tensorboard", name="ATSS_detector_train"
    )

    print(f'train ATSS_detector_train')
    num_epochs = cfg.DATALOADER.NUM_EPOCHS
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=tb_logger,
        gpus=None, 
    )

    trainer.fit(model, train_loader, valid_loader)
    return trainer


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = train(cfg)

if __name__ == "__main__":
    main()
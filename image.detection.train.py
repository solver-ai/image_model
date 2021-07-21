
import argparse
import os

# import torch

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from imre.atss.configs.default import _C as cfg


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



from imre.dataset.detection_datasets import COCODataset

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

#### check ####
from imre.model.pl_atss import ATSSModel
import torch 
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from imre.module.utils import BatchCollator


def train(cfg):

    train_dataset, valid_dataset = get_dataset() #data_path=cfg.DATA_PATH)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    
    collator = BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)

    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collator,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collator,
    )

    # fearture learning preparation
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model = train(cfg)

if __name__ == "__main__":
    main()
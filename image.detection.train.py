
import argparse
import os

# import torch

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

IMG_TRANSFORMS_PIPELINE = {
    "train" : A.Compose([
        A.LongestMaxSize(512),
        A.PadIfNeeded(
            min_height=512,
            min_width=512, 
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
            min_height=512,
            min_width=512, 
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
        ann_file='datasets/deepfashion2/train.json',
        root='datasets/deepfashion2/train',
        transforms=IMG_TRANSFORMS_PIPELINE['train']
    )
    valid_dataset = COCODataset(
        ann_file='datasets/deepfashion2/valid.json', 
        root='datasets/deepfashion2/valid',
        transforms=IMG_TRANSFORMS_PIPELINE['valid']
    )
    return train_dataset, valid_dataset

#### check ####
from imre.module import utils
from imre.model.pl_atss import ATSSModel
import torch 
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl



def train(cfg):

    train_dataset, valid_dataset = get_dataset() #data_path=cfg.DATA_PATH)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    batch_size = cfg.SOLVER.IMS_PER_BATCH

    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=utils.collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=utils.collate_fn,
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




#### check ####
'''
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
    )

    return model

def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
'''

from imre.atss.configs.default import _C as cfg
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    cfg.merge_from_file('imre/atss/configs/atss_R_101_FPN_2x.yaml')
    cfg.freeze()
    print(cfg)

    model = train(cfg)


if __name__ == "__main__":
    main()

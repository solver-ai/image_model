"""
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch
"""
import os
from imre.model.pl_resnet import ResnetModel

import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from pytorch_lightning import loggers as pl_loggers


IMG_TRANSFORMS_PIPELINE = {
    "train" : [
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
    ],
    "valid" : [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.5,.5,.5],[.5,.5,.5]),
    ]
}

## grid search
"""
SEARCH_SPACE = {
    "loss": tune.choice(["multisim"]),
    "pool": tune.choice(["avg", "rmac", "gem"]),
    "norm_pool": tune.choice([True, False]),
    "trunk_lr": tune.choice([0.00005, 0.00001]),
    "embedder_lr": tune.choice([0.01, 0.005, 0.001]),
    "clf_lr": tune.choice([0.01, 0.005, 0.001]),
    "m_per_class": tune.choice([3, 5]),
    "batch_size": 64,
    "num_epochs": 1,
    "efficientnet_model_name": "efficientnet-b3",
}
"""

BEST_CONFIG = {
    "loss": "CrossEntropy",
    "pool": "ang",
    "norm_pool": True,
    "base_lr": 3e-2,
    "batch_size": 4,
    "num_epochs": 20,
    "model": "resnet101"
}

def get_dataset(data_path: str="datasets"):
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_path,"train"),
        transform=transforms.Compose(IMG_TRANSFORMS_PIPELINE["train"]),
    )
    valid_dataset = datasets.ImageFolder(
        root=os.path.join(data_path, "valid"),
        transform=transforms.Compose(IMG_TRANSFORMS_PIPELINE["valid"]),
    )
    return train_dataset, valid_dataset

def train(config, mode:str ="tune"):
    args = config
    print(args)

    ##dataset

    train_dataset, valid_dataset= get_dataset()

    num_workers = 0
    batch_size = args.get("batch_size")

    # data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # fearture learning preparation
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = ResnetModel(config)

    tb_logger = pl_loggers.TensorBoardLogger(
        "artifacts/tensorboard", name="classification_test"
    )

    print("train resnet model")
    num_epochs = args.get("num_epochs")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=tb_logger,
        gpus=None,
    )

    trainer.fit(model, train_loader, valid_loader)

    # if mode =="tune":
    #     tune.report(map=)
    # else:
    return trainer

if __name__ == "__main__":
    import easydict
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    if args.mode == "train":
        cfg = easydict.EasyDict(BEST_CONFIG)
        tr = train(cfg, mode="train")
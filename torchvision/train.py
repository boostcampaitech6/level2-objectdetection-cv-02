import argparse
import collections
import torch
import numpy as np
import os
import random
from importlib import import_module
from torch.utils.data import DataLoader
import dataloader.datasets as module_dataset
import dataloader.augmentations as module_augmentation
import model.loss as module_loss
import model.model as module_arch
from trainer import Trainer
from torch.optim.lr_scheduler import StepLR



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def collate_fn(batch):
    return tuple(zip(*batch))


def main(data_dir, model_dir, config):
    seed_everything(config.seed)

    # settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # setup augmentation instance
    augmentation_module = getattr(module_augmentation, config.augmentation)  # default: CustomAugmentation
    transform = augmentation_module(
        resize=config.resize,
    )
    
    # setup data_set instance
    dataset_module = getattr(module_dataset, config.dataset)  # default: CustomNormalDataset
    train_dataset = dataset_module(
        annotation=config.annotation,
        data_dir=config.data_dir,
        transforms=transform
    )

    # setup data_loader instances
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    # build model architecture, then print to console
    model_module = getattr(module_arch, config.model)
    model = model_module(num_classes=11).to(device)
    model = torch.nn.DataParallel(model)

    # get function handles of loss and metrics
    criterion = module_loss.create_criterion(config.criterion)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer_module = getattr(import_module("torch.optim"), config.optimizer)  # default: Adam
    optimizer = optimizer_module(
        trainable_params,
        lr=config.lr,
    )
    lr_scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    trainer = Trainer(model, criterion, optimizer,
                      config=config,
                      device=device,
                      train_dataloader=train_dataloader,
                      valid_dataloader=None,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 64)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="CustomNormalDataset",
        help="dataset augmentation type (default: CustomNormalDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="CustomAugmentation",
        help="data augmentation type (default: CustomAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[1024, 1024],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=100,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="fasterrcnn_resnet50_fpn", help="model type (default: fasterrcnn_resnet50_fpn)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: Adam)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )
    parser.add_argument(
        "--wandb", default="model_fasterrcnn_resnet50_fpn", 
        help="wandb run name. 실험 대상이 되는 \"arg종류_arg값\" 형태로 적어주세요 (예: model_fasterrcnn_resnet50_fpn)."
    )


    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/data/ephemeral/dataset/train/")
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/data/ephemeral/model")
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default=os.environ.get("SM_annot", "/data/ephemeral/dataset/train.json")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    main(data_dir, model_dir, args)

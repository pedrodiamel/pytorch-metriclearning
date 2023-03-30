import datetime
import os
import random

import numpy as np

# TORCH MODULE
import torch.backends.cudnn as cudnn

# PYTVISION MODULE
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler

# TORCHVISION MODULE
from torchvision import transforms

# LOCAL MODULES
from .configs.train_config import AugmentationConfig, Config
from .datasets import TripletsDataset
from .datasets.factory import FactoryDataset
from .neuralnet import NeuralNetTriplet


def remove_files(path):
    for root, _, files in os.walk(path):
        for f in files:
            file_path = os.path.join(root, f)
            if os.path.isfile(file_path):
                os.remove(file_path)


def train(cfg: Config, aug: AugmentationConfig):

    print("NN Clasification {}!!!".format(datetime.datetime.now()))
    random.seed(cfg.seed)

    project_path = os.path.expanduser(cfg.project.path)
    if os.path.isdir(project_path) is not True:
        os.makedirs(project_path)

    project_name = "{}_{}_{}_{}_{}_v{}".format(
        cfg.project.name_prefix,
        cfg.trainer.arch,
        cfg.trainer.loss,
        cfg.trainer.opt,
        cfg.data.name_dataset,
        cfg.project.version,
    )

    # Check path project
    # TODO February 18, 2023: Check this function for include in pytvision
    project_pathname = os.path.expanduser(os.path.join(project_path, project_name))
    if os.path.exists(project_pathname) is not True:
        os.makedirs(project_pathname)
    else:
        response = input("Do you want to remove all files in this folder? (y/n): ")
        if response.lower() == "y":
            remove_files(project_pathname)

    # Create and load neural net training class
    # TODO February 18, 2023: I think that we can input the cfgs
    network = NeuralNetTriplet(
        patchproject=project_path,
        nameproject=project_name,
        no_cuda=not cfg.trainer.cuda,
        parallel=cfg.trainer.parallel,
        seed=cfg.seed,
        print_freq=cfg.checkpoint.print_freq,
        gpu=cfg.trainer.gpu,
    )

    network.create(
        arch=cfg.trainer.arch,
        num_output_channels=cfg.trainer.dim,
        num_input_channels=cfg.trainer.numchannels,
        loss=cfg.trainer.loss,
        lr=cfg.trainer.lr,
        momentum=cfg.trainer.momentun,
        optimizer=cfg.trainer.opt,
        lrsch=cfg.trainer.scheduler,
        pretrained=cfg.trainer.finetuning,
        size_input=cfg.trainer.image_size,
        classes=cfg.trainer.numclass,
        margin=cfg.trainer.margin
    )

    # Set cuda cudnn benchmark true
    cudnn.benchmark = True

    # Resume model
    if cfg.checkpoint.resume:
        network.resume(os.path.join(network.pathmodels, cfg.checkpoint.resume))

    print("Load model: ")
    # print(network)

    # Load dataset
    size_input = cfg.trainer.image_size

    # Load training dataset
    train_data = TripletsDataset(
        data=FactoryDataset.factory(
            pathname=cfg.data.path,
            name=FactoryDataset.str_to_dataset[cfg.data.name_dataset],
            subset=FactoryDataset.Subsets.TRAIN,
            download=True,
        ),
        n_triplets=cfg.data.num_triplet_train,
        num_channels=network.num_input_channels,
        transform=transforms.Compose(aug.transforms_train(size_input)),
    )

    num_train = len(train_data)
    sampler = SubsetRandomSampler(np.random.permutation(num_train))

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.data.batch_size,
        sampler=sampler,
        num_workers=cfg.data.workers,
        pin_memory=network.cuda,
        drop_last=True,
    )

    # Load validation dataset
    val_data = TripletsDataset(
        data=FactoryDataset.factory(
            pathname=cfg.data.path,
            name=FactoryDataset.str_to_dataset[cfg.data.name_dataset],
            subset=FactoryDataset.Subsets.VAL,
            download=True,
        ),
        n_triplets=cfg.data.num_triplet_val,
        num_channels=network.num_input_channels,
        transform=transforms.Compose(aug.transforms_val(size_input)),
    )

    num_val = len(val_data)
    sampler = SubsetRandomSampler(np.random.permutation(num_val))

    val_loader = DataLoader(
        val_data,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        sampler = sampler,
        num_workers=cfg.data.workers,
        pin_memory=network.cuda,
        drop_last=False,
    )

    print("Load datset")
    print("Train: ", len(train_data))
    print("Val: ", len(val_data))

    # training neural net
    network.fit(train_loader, val_loader, cfg.data.epochs, cfg.checkpoint.snapshot)

    print("Optimization Finished!")
    print("DONE!!!")

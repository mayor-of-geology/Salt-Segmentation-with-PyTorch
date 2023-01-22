
from pathlib import Path
from os import cpu_count
from typing import Tuple
from get_data import data
from segdataset import SaltSegmentationDataset
from augmentation import train_augs, test_augs

from torch.utils.data import DataLoader

import torch

def create_loaders(DATA_PATH:Path, batch_size:int) -> Tuple[DataLoader, DataLoader]:
    
    traindata, testdata = data(DATA_PATH)
    
    #load dataset and apply augmentation techniques
    trainset = SaltSegmentationDataset(traindata.images, traindata.masks, train_augs())
    testset = SaltSegmentationDataset(testdata.images, testdata.masks, test_augs())

    #WRAP AN iterable dataloader around the train images and masks
    trainloader = DataLoader(trainset, batch_size=batch_size, 
                             pin_memory=True if torch.cuda.is_available() else False , shuffle=True, num_workers=cpu_count())
    testloader = DataLoader(testset, batch_size=batch_size, 
                            pin_memory=True if torch.cuda.is_available() else False , num_workers=cpu_count())
    
    return trainloader, testloader

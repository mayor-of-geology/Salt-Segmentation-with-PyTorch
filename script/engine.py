
from typing import Tuple, List
from numpy import Inf
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import device
from tqdm.auto import tqdm

from script.architecture import SaltSegmentationModel

import torch

def train_func(dataloader:DataLoader, model:SaltSegmentationModel, optimizer:Adam) -> float:
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loss = 0.0
    
    model.train()
    
    for images, masks in tqdm(dataloader):

        images = images.permute(0, 3, 1, 2); masks = masks.permute(0, 3, 1, 2)
        images, masks = images.to(DEVICE), masks.to(DEVICE)
        
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
    return train_loss / len(dataloader)


def test_func(dataloader:DataLoader, model:SaltSegmentationModel) -> float:
    
    model.eval()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    test_loss = 0.0
    with torch.inference_mode():
        for images, masks in tqdm(dataloader):
            images = images.permute(0, 3, 1, 2); masks = masks.permute(0, 3, 1, 2)
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            
            logits, loss = model(images, masks)
            
            test_loss += loss.item()
            
    return test_loss / len(dataloader)


def train(trainloader:DataLoader, testloader:DataLoader, model:SaltSegmentationModel,
          optimizer:Adam, EPOCHS:int, DIR:Path) -> Tuple[List, List]:
    
    best_valid_loss = Inf

    print("*"*60)
    print('                         START TRAINING               ')
    print("*"*60)
    train_losses, test_losses = list(), list()
    for i in tqdm(range(EPOCHS)):
        train_loss = train_func(trainloader, model, optimizer)
        test_loss = test_func(testloader, model)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if test_loss < best_valid_loss:
            torch.save(model.state_dict(), DIR/'model/best_salt_model.pt')
            print('SAVED MODEL')

            best_valid_loss = test_loss

        print(f'Epoch : {i+1} : Training loss : {train_loss} | Validation loss : {test_loss}')

    print("*"*60)
    print('                         TRAINING ENDS               ')
    print("*"*60)
    
    return train_losses, test_losses


from torchvision import io
from torch.utils.data import Dataset
from typing import Tuple
from pathlib import Path
import torch
import cv2

class SaltSegmentationDataset(Dataset):
    def __init__(self, imagePaths:Path, maskPaths:Path, augmentations) -> Tuple[torch.Tensor, torch.Tensor]:
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.augmentations = augmentations
#         self.train = train
        
        
    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        
        image = io.read_image(str(self.imagePaths[idx]), mode=io.ImageReadMode.GRAY).permute(1, 2, 0).numpy()
        mask = torch.tensor(cv2.cvtColor(cv2.imread(str(self.maskPaths[idx]), 1), cv2.COLOR_BGR2GRAY)).unsqueeze(2).numpy()
        
#         augment = DataFrame()
        
        augment = self.augmentations(image=image, mask=mask)            
        image = augment['image']
        mask = augment['mask']
        
        image = torch.Tensor(image).type(torch.float32)/255.0
        mask = torch.Tensor(mask).type(torch.float32)/255.0
        
        return image, mask

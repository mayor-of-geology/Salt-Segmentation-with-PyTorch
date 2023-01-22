
from torch import nn
from segmentation_models_pytorch.losses import DiceLoss
import segmentation_models_pytorch as smp

class SaltSegmentationModel(nn.Module):
    
    def __init__(self):
        super(SaltSegmentationModel, self).__init__()
        
        self.arc = smp.Unet(
            encoder_name='timm-efficientnet-b0',
            encoder_weights='imagenet',
            in_channels=1,
            classes=1,
            activation=None
        )
        
    def forward(self, images, masks=None):
        logits = self.arc(images)
        
        if masks != None:
            dice_loss = DiceLoss(mode='binary')(logits, masks)
            bce_loss = nn.BCEWithLogitsLoss()(logits, masks)
            
            return logits, dice_loss + bce_loss
    
        return logits

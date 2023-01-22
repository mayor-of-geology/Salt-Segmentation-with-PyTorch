
import albumentations as A

def train_augs(prob:int=1) -> A.Compose:
    return A.Compose([
            A.Resize(128, 128),
            A.PadIfNeeded(min_height=128, min_width=128, p=prob),
            A.CropNonEmptyMaskIfExists(width=128, height=128, p=prob),
            A.Superpixels(max_size=128),
#             A.Normalize(),
        ])

def test_augs() -> A.Compose:
    return A.Compose([
            A.Resize(128, 128),
#             A.Normalize()
        ])

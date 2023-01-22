
import torch
from torchvision import io
from argparse import ArgumentParser
from pathlib import Path
from script.architecture import SaltSegmentationModel
from torchvision.transforms import Resize
from matplotlib import pyplot as plt

parser = ArgumentParser()

parser.add_argument('--image_path', type=str, help='Image to predict mask')

parser.add_argument('--model_path', 
                    default=Path('./model/best_salt_model.pt'), type=str,
                    help='Model Path')

parser.add_argument('--image_name', type=str, help='Image to predict mask')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(filepath=args.model_path):
  
    model = SaltSegmentationModel().to(device)                             
    model.load_state_dict(torch.load(filepath))

    return model

def predict_mask(image_path=args.image_path, filepath=args.model_path):

    model = load_model(filepath)

    image = io.read_image(str(image_path), mode=io.ImageReadMode.GRAY).numpy()#.permute(1, 2, 0).numpy()

    image = torch.Tensor(image).type(torch.float32)/255.0

    # Resize the image to be the same size as the model
    transform = Resize(size=(128, 128))
    image = transform(image)#.squeeze(dim=0) 

    # Predict on image
    model.eval()
    with torch.inference_mode():

        image = image.unsqueeze(1).to(device)
        model.to(device)

        logits = model(image)
        pred_mask = (torch.sigmoid(logits).type(torch.float32)) > 0.5 * 1.0

    return image, pred_mask

def plot_mask():

    image, pred_mask = predict_mask(image_path=args.image_path, filepath=args.model_path)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    ax1.set_title('IMAGE')
    ax1.imshow(image.detach().cpu().squeeze(), cmap='seismic_r')

    ax2.set_title('PREDICTED')
    ax2.imshow(pred_mask.detach().cpu().squeeze(), cmap='jet')

    plt.savefig('./pictures/'+args.image_name+'.png')

if __name__ == '__main__':
    plot_mask()
     

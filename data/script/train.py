
import os

from argparse import ArgumentParser
from pathlib import Path
from script.dataloader import create_loaders
from script.architecture import SaltSegmentationModel
from script.engine import train
from torch.optim import Adam
from matplotlib import pyplot as plt

import torch

parser = ArgumentParser(description='Get some hyperparameters', add_help=True)

parser.add_argument('-e', '--num_epochs', default=5, metavar='EPOCH', type=int, help='number of epochs')
parser.add_argument('-bs', '--batch_size', default=32, type=int, help='number of batch or sample size')
parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, metavar='LR', help='learning rate')
parser.add_argument('--data-dir', default=Path('./'), help='directory of training data', type=Path)

args = parser.parse_args()

#hyperparameters
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
LR = args.learning_rate

#data directory
DIR = args.data_dir
print(f'Data file path : {DIR}')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader = create_loaders(DIR, BATCH_SIZE)

model = SaltSegmentationModel().to(device)

optimizer = Adam(model.parameters(), lr=LR)

def plot_loss():
    train_losses, test_losses = train(trainloader, testloader, model, optimizer=optimizer, EPOCHS=NUM_EPOCHS, DIR=DIR)

    # plot the training loss
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.plot(test_losses, label="test_loss")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.savefig(DIR/'pictures/losses.png')

if __name__ == '__main__':
    plot_loss()

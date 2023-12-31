import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import wandb

wandb.init()

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        plt.savefig(f'{title}.png')
        wandb.log(
            {wandb.Image(Image.open(f'{title}.png'), caption=f"{title}")}
        )
    plt.savefig('result.png')
    wandb.log(
            {wandb.Image(Image.open('result.png'), caption="result")}
        )
    plt.pause(0.001)  # pause a bit so that plots are updated


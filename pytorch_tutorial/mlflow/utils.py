import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import mlflow 

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
        plt.savefig(f'{title}.jpg')
        # mlflow.log_image(Image.open(f'{title}.jpg'), f"{title}")
    plt.savefig('result.jpg')
    # mlflow.log_image(Image.open('result.jpg'), "result")
    plt.pause(0.001)  # pause a bit so that plots are updated


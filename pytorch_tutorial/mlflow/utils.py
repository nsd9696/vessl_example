import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import vessl

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
        vessl.log(
            payload={
                f"{title}": [vessl.Image(data=Image.open(f'{title}.png'), caption=f"{title}")]
            }
        )
    plt.savefig('result.png')
    vessl.log(
        payload={
            "result": [vessl.Image(data=Image.open('result.png'), caption="result")]
        }
    )
    plt.pause(0.001)  # pause a bit so that plots are updated


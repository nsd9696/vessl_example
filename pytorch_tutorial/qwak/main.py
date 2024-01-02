import qwak
from qwak import QwakModel
import torch

class MyModel(QwakModel):
    def __init__(self):
        self._gamma = 'scale'
 
    def build(self):
        self.model = torch.load("torch_model.pt")
        self.model = self.model.eval()
 
    def predict(self, img):
        with torch.no_grad():
            outputs = model(img)
        return outputs
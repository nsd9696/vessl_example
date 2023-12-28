import matplotlib.pyplot as plt
from train import finetune_resnet
from inference import visualize_model_predictions
from utils import visualize_model

model = finetune_resnet()
visualize_model(model_conv)
visualize_model_predictions(
    model_conv,
    img_path='data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

plt.show()
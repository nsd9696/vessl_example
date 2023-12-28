import matplotlib.pyplot as plt
from pytorch_tutorial.train_inference import finetune_resnet
from train_inference import visualize_model_predictions
from utils import visualize_model

model = finetune_resnet()
visualize_model(model)
visualize_model_predictions(
    model,
    img_path='data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

plt.show()
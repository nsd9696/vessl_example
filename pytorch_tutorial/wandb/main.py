import matplotlib.pyplot as plt
from train_inference import finetune_resnet
from train_inference import visualize_model_predictions
from train_inference import visualize_model
import wandb

wandb.login()
wandb.init()

model = finetune_resnet()
visualize_model(model)
visualize_model_predictions(
    model,
    img_path='../../../data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

wandb.save("torch_model.pt")

# best_model = wandb.Artifact(f"model_{run.id}", type="model")
# best_model.add_file("my_model.h5")

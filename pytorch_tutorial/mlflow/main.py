import matplotlib.pyplot as plt
from train_inference import finetune_resnet
from train_inference import visualize_model_predictions
from train_inference import visualize_model

model = finetune_resnet()
visualize_model(model)
visualize_model_predictions(
    model,
    img_path='../../data/hymenoptera_data/val/bees/72100438_73de9f17af.jpg'
)

# vessl.create_model_repository
# vessl.create_model
# vessl.upload_model_volume_file
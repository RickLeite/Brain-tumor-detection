import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms as T

device = torch.device('cpu')

# Load the model
model_path = "modelo_segmentacao.pth"
model = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation='sigmoid',
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

def apply_segmentation_mask(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    target_height = 224
    target_width = 192
    resized_image = cv2.resize(image, (target_width, target_height))
    image_tensor = T.ToTensor()(resized_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        prediction = model(image_tensor).cpu()
        binary_mask = prediction.squeeze().numpy()
        binary_mask = np.where(binary_mask > 0.5, 1, 0)

    # Overlay the mask on the original image
    overlayed_image = resized_image.copy()
    overlayed_image[binary_mask == 1] = [255, 0, 0]

    return overlayed_image

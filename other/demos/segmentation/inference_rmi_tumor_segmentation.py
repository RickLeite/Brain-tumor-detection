import torch
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms as T
import numpy as np

device = torch.device('cpu')

# model architecture
model = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights=None,
    in_channels=3,
    classes=1,
    activation='sigmoid',
).to(device)

# Load the model
model_path = "modelo_segmentacao.pth"
model.load_state_dict(torch.load(model_path, map_location=device)) # load_state_dict applies the loaded parameters to the model (weights, biases, and other parameters)
model.eval() # eval() will notify all your layers that you are in eval mode
             # that way, batchnorm or dropout layers will work in eval mode instead of training mode

# Load and preprocess the image
image_path = "data/TCGA_FG_6691_20020405_21.tif"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

target_height = 224
target_width = 192
resized_image = cv2.resize(image, (target_width, target_height))

# Convert image (NumPy Array) into PyTorch tensor (multi-dimensional arrays that effienctly managed by GPU)
image_tensor = T.ToTensor()(resized_image).unsqueeze(0).to(device)

# Perform inference
with torch.no_grad(): # we don't need to compute gradients for backpropagation when doing inference
    prediction = model(image_tensor).cpu() # probably shape (batch_size, channels, height, width)
    # output of the the prediction is a tensor representing the segmentation mask, 
    # where each pixel value indicates the probability of that pixel belonging to the foreground class 

# Apply a threshold to get a binary mask
threshold = 0.5 # pixels below the threshold are classified as background, and above as foreground
binary_mask = prediction.squeeze().numpy() # Tensor (batch_size, channels, height, width) to (height, width) NumPy array
binary_mask = np.where(binary_mask > threshold, 1, 0) # True False to 1 0

# Overlay the mask on the original image
# Note: You might need to adjust the color and transparency of the mask
overlayed_image = resized_image.copy()
overlayed_image[binary_mask == 1] = [255, 0, 0]  # Red color where pixels are classified as foreground (1)

# Convert overlayed image back to BGR for saving with OpenCV
overlayed_image_bgr = cv2.cvtColor(overlayed_image, cv2.COLOR_RGB2BGR)

# Saving the resulting image
result_path = "data/result_with_mask.jpg"
cv2.imwrite(result_path, overlayed_image_bgr)

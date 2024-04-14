import requests
from io import BytesIO

url = "http://127.0.0.1:8000/segmentation/"

image_path = "TCGA_DU_A5TW_19980228_18.tif"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

if response.status_code == 200:

    segmented_image_data = response.content

    with open("segmented_image.jpg", "wb") as f:
        f.write(segmented_image_data)


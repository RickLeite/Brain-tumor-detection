import os
import requests
from io import BytesIO

files = os.listdir()

tif_files = [file for file in files if file.endswith('.tif')]

print("Please select a file:")
for i, file in enumerate(tif_files):
    print(f"{i+1}. {file}")

selection = int(input("Enter the number of your selection: ")) - 1
image_path = tif_files[selection]

url = "http://127.0.0.1:8000/segmentation/"

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

if response.status_code == 200:
    segmented_image_data = response.content
    with open("segmented_image.jpg", "wb") as f:
        f.write(segmented_image_data)
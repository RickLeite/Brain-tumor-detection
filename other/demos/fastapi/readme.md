# Building the image
docker build -t my_fastapi_app .

# Running Docker Container interactively and mounting volume  
docker run -it -p 8000:8000 -v ./volume:/app/output my_fastapi_app /bin/bash

# Running the FastAPI Server
uvicorn main:app --host 0.0.0.0

# Sending POST Request
curl -X POST -F "file=@TCGA_DU_A5TW_19980228_18.tif" http://localhost:8000/segmentation/ --output segmented_image.jpg

or with send_post.py file (running out of container)


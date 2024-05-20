## Building the image
docker build -t backend_inference_api .

## Running Docker
docker run -p 8000:8000 -v ./output:/app/output backend_inference_api uvicorn app.main:app --host 0.0.0.0 --port 8000

## Test Sending POST Request with Image:
You can test it by sending a POST request with an image file by using the [send_request.py](./local_tests/send_request.py) file. The image with the mask will be saved in the `output` folder.

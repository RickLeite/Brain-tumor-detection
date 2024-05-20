from fastapi import FastAPI, Request, HTTPException
from PIL import Image
from fastapi.responses import FileResponse
import io
from inference import apply_segmentation_mask
import os
import uuid
import numpy as np

app = FastAPI()

OUTPUT_DIR = "/app/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/segmentation/")
async def segment_image(request: Request):
    try:
        contents = await request.body()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        segmented_image = apply_segmentation_mask(np.array(image))

        segmented_image_pil = Image.fromarray(segmented_image.astype('uint8'))

        filename = f"{uuid.uuid4()}.png"
        segmented_image_path = os.path.join(OUTPUT_DIR, filename)

        segmented_image_pil.save(segmented_image_path)

        return FileResponse(segmented_image_path, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
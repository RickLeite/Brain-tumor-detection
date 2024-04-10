## Streamlit App

Streamlit app (Proof-of-Concept) for running the inference of segmentation model.

1. **Build the Docker image:**

    Use the following command to build the Docker image for the Streamlit app:

    ```sh
    docker build -t streamlit-app .
    ```

2. **Run the Docker container:**

    After building the image, you can run the Docker container. This command also sets up a volume for output images with masks and forwards port 8501:

    ```sh
    docker run -v /path-to-your-volume:/app/volume -p 8501:8501 -it streamlit-app bash
    ```

3. **Run the Streamlit app:**

    Once inside the Docker container, you can start the Streamlit app with the following command:

    ```sh
    streamlit run main.py
    ```

4. **Access the Streamlit app:**

    After starting the Streamlit app, you can access it by visiting [http://localhost:8501/](http://localhost:8501/) in your web browser.

5. **Perform Inference:**

    Now you can use the Streamlit app to perform inference with the models. Enjoy!
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import os
from inference import apply_segmentation_mask

def save_uploaded_image(uploaded_file):
    save_path = "./volume/imageraw.tif"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

def load_image_with_mask():
    uploaded_file = st.file_uploader(label='Escolha a imagem', type=['jpg', 'jpeg', 'png', 'tif'])
    if uploaded_file is not None:
        # Save the uploaded image
        saved_image_path = save_uploaded_image(uploaded_file)

        # Apply segmentation mask
        masked_image = apply_segmentation_mask(saved_image_path)

        # Display images side by side
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.header("Imagem Original")
                st.image(saved_image_path, use_column_width=True)
            with col2:
                st.header("Máscara Segmentação")
                st.image(masked_image, use_column_width=True)

def main():
    st.title('Segmentação Tumor Cerebral')
    load_image_with_mask()

if __name__ == '__main__':
    main()

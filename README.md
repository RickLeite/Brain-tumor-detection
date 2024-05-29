### Brain MRI Segmentation with UNet and EfficientNetB7

This project focuses on the segmentation of brain MRI images using a UNet model with EfficientNetB7 as the backbone. The dataset used in this project is the LGG Segmentation Dataset from Kaggle, which includes brain MRI images and manual FLAIR abnormality segmentation masks.

## Dataset

The dataset is based on the following Kaggle dataset:
[Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

The LGG Segmentation Dataset contains brain MRI images together with manual FLAIR abnormality segmentation masks. The images were obtained from [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/) and correspond to 110 patients included in [The Cancer Genome Atlas (TCGA)](https://www.genome.gov/Funded-Programs-Projects/Cancer-Genome-Atlas) lower-grade glioma collection with at least fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.

### Publications

The dataset has been used in the following publications:

1. Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski. "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019.

2. Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, Katherine B. Peters, Ashirbani Saha. "Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.

## Model

The notebook demonstrates the training of a segmentation model using UNet architecture with EfficientNetB7 as the backbone. The choice of EfficientNetB7 enhances the feature extraction capabilities of the UNet model, making it well-suited for medical image segmentation tasks.

### Model Architecture

- **UNet**: A convolutional neural network architecture designed for biomedical image segmentation.
- **EfficientNetB7**: A high-performance convolutional neural network that serves as the encoder (backbone) in the UNet model. Pre-trained on ImageNet, it provides robust feature extraction.

### Model Implementation

The model implementation is based on and provided to the community by [Abdullah Wagih](https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-efficientnetb7/notebook), an AI Engineer.

## Project Structure

- **base**: Contains everything related to the model architecture and training.
    - **data/**: Contains the dataset CSV file (image data not included).
    - **notebook/**: Contains the Jupyter notebook for training the model.
    - **models/**: Directory with the saved trained model.
- **backend_inference_api**: Contains the API developed with FastAPI for the model inference.
    - **app/**: Contains the main files for the API.
    - **models/**: Directory with the saved trained model.
    - **requirements.txt**: Required libraries for the API.
- **program**: Contains the program created with Java Spring and Postgres for integration.
    - [Projeto Integrador V](https://github.com/josedurigon/ProjetoIntegradorV): GitHub repository with the integration code.

## Training

The training process includes the following steps:

1. **Data Preprocessing**: Loading and augmenting the data.
2. **Model Training**: Training the UNet model with EfficientNetB7 backbone.
3. **Evaluation**: Evaluating the model on the validation set using metrics like IoU and Dice coefficient.
4. **Early Stopping**: Implementing early stopping to prevent overfitting.

### Example Code
#### Model Definition

```python
model = smp.Unet(
    encoder_name="efficientnet-b7",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1,
    activation='sigmoid',
)
```

## Program Integration

The integration project, presented on May 27, 2024, as part of an extension course, demonstrates the seamless connection between the trained model, the API for inference, and a comprehensive Java Spring application for medical professionals. This system was showcased to a panel of professors, Saullo Oliveira and Dimas Augusto, highlighting its functionality and potential in a clinical setting.

### Integration Details

- **API Development**: The FastAPI-based backend inference API allows for real-time segmentation predictions on new MRI scans. It ensures a quick and reliable interface between the trained model and the Java application.
- **Java Spring Application**: The Java Spring application, coupled with a Postgres database, provides a user-friendly interface for medical professionals to upload MRI images, retrieve segmentation results, and visualize possible abnormalities.
### System Workflow

1. **Patient Registration**: The system includes a patient registration feature.
2. **Image Upload**: Medical professionals can upload MRI images through the user-friendly interface of the Java Spring application.
3. **Inference Request**: Once uploaded, the MRI images are automatically sent to the FastAPI backend for processing.
4. **Segmentation Results**: The FastAPI backend utilizes the trained UNet model to analyze the MRI images and generate segmentation masks, identifying potential abnormalities.
5. **Visualization**: The segmentation results are seamlessly integrated back into the Java Spring application, where they are displayed for doctors to review and analyze, facilitating better diagnostic decisions.

<br><br>
[![IA MRI MEGACARE SYSTEM](./images/megacare_preview.jpeg)](https://github.com/josedurigon/ProjetoIntegradorV/tree/main/src/main/java/com/example/demo)


## Conclusion

This project demonstrates the effectiveness of using UNet with EfficientNetB7 for brain MRI segmentation. By leveraging state-of-the-art deep learning techniques, we achieve robust segmentation performance, aiding in the analysis and diagnosis of abnormalities. The integration with a user-friendly application further enhances its practical applicability, providing a valuable tool for medical professionals.

---

### References

- [Journal of Neuro-Oncology, 2017](https://pubmed.ncbi.nlm.nih.gov/28470431/)
- [Computers in Biology and Medicine, 2019](https://arxiv.org/abs/1906.03720)
- [Brain MRI Segmentation Dataset on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
- [Abdullah Wagih's Notebook](https://www.kaggle.com/code/abdallahwagih/brain-tumor-segmentation-unet-efficientnetb7/notebook)
- [Projeto Integrador V GitHub Repository](https://github.com/josedurigon/ProjetoIntegradorV)
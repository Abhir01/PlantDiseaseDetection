# Real-Time Tomato Leaf Disease Detection

## Overview

The "Real-Time Plant Disease Detection" project aims to assist farmers in accurately identifying diseases in tomato plants through a user-friendly mobile application. 
By leveraging Convolutional Neural Networks (CNNs) and deep learning techniques, the application can classify various tomato plant diseases from images taken by the farmer's mobile device. 
The model has achieved an accuracy of 96.44%, ensuring reliable disease detection.

## Features

- **Real-time Disease Detection**: Capture an image of a tomato plant leaf and get an instant diagnosis.
- **High Accuracy**: The CNN model used achieves an accuracy of 96.44%.
- **User-Friendly Interface**: Intuitive mobile application built using React Native.
- **Scalability**: Deployed on Google Cloud Platform to handle large-scale usage.

## Technologies Used

- **Backend**:
  - TensorFlow
  - TensorFlow Lite
  - FastAPI
  - Google Cloud Platform (GCP)

- **Frontend**:
  - React Native
  - React Js(for web app)

- **Model**:
  - Convolutional Neural Network (CNN)

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- Node.js
- TensorFlow
- FastAPI
- Google Cloud SDK

### Clone the Repository

```bash
git clone https://github.com/yourusername/plant-disease-detection.git
cd plant-disease-detection
```

### Backend Setup

1. **Create and Activate a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run FastAPI Server**

    ```bash
    uvicorn main:app --reload
    ```

### Frontend Setup

1. **Install Dependencies**

    ```bash
    cd frontend
    npm install
    ```

2. **Run React Native Application**

    ```bash
    npm start
    ```

## Model Training

### Dataset

The dataset used for training the CNN model consists of images of tomato plant leaves categorized into different disease classes. The images are preprocessed and augmented to enhance the training process.

### Training Script

1. **Run the training script**

    ```bash
    python train_model.py
    ```

### Model Evaluation

The model is evaluated using a separate validation set to ensure its accuracy and robustness. The achieved accuracy is 96.44%.

### Exporting Model

The trained model is exported and converted to TensorFlow Lite format for deployment.

```python
import tensorflow as tf

model = tf.keras.models.load_model('path_to_model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

## Deployment

The TensorFlow Lite model is deployed on Google Cloud Platform, and Google Cloud Functions are used to serve the model for predictions.

### Deploy to GCP

1. **Upload the Model to GCP**

2. **Create Google Cloud Functions**

3. **Link the Mobile App to Cloud Functions**

## Usage

1. **Open the Mobile App**

2. **Capture an Image of a Tomato Leaf**

3. **Receive Diagnosis**


For any questions or suggestions, please contact:

- Name: Abhishek Rawat
- Email: abr07k@gmail.com
- GitHub: github.com/Abhir01/

---


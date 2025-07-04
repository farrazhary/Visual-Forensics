# Visual Forensics: Deep Learning for Insurance Fraud Detection

## The project contains files as below
```
1. Visual Forensics.ipynb: A notebook containing the full training pipeline, including data preprocessing, model architecture, training, and evaluation.
2. Visual Forensics Inference.ipynb: A notebook for performing inference using the trained model on new or unseen vehicle damage images.
```

## Problem Background
nsurance fraud often involves submitting false claims or exaggerating the extent of damage in an attempt to receive higher compensation. Such fraudulent activities pose a significant financial threat to insurance companies and compromise the integrity of the claims process. To mitigate this, there is a growing need to implement intelligent, scalable, and automated systems that can accurately differentiate between genuine and fake vehicle damage claims. Leveraging computer vision and deep learning, this project aims to enhance the fraud detection process by analyzing images of reported vehicle damages.

## Project Output
This project produces a deep learning model capable of classifying vehicle damage images as fraudulent or legitimate. The notebook includes the complete pipeline: data loading, image preprocessing, model training using transfer learning, model evaluation, and prediction on unseen data. The output also includes performance metrics to assess the model's reliability in identifying fake claims.

## Dataset Description
The [dataset](https://www.kaggle.com/datasets/pacificrm/car-insurance-fraud-detection) consists of labeled vehicle damage images, categorized into genuine and fraudulent claims. Each image provides visual evidence of the reported damage, which serves as the input for the computer vision model. The dataset has been preprocessed and augmented to ensure a balanced distribution between classes and improve the model's generalization.

## Analysis Method
The project utilizes a computer vision pipeline built on deep learning, particularly convolutional neural networks (CNNs) and transfer learning. The process includes image augmentation, model training, validation, and threshold tuning based on performance metrics on F1 Score.

## Library
This project utilizes a comprehensive set of Python libraries to support computer vision, deep learning, data preprocessing, and evaluation workflows.

For deep learning and model training, the project uses TensorFlow and Keras, including advanced architectures such as ConvNeXtLarge, InceptionResNetV2, and DenseNet201. Key layers such as Conv2D, GlobalAveragePooling2D, BatchNormalization, and callbacks like EarlyStopping, ModelCheckpoint, and ReduceLROnPlateau are used to optimize training. The ImageDataGenerator class is applied for real-time image augmentation.

Image processing tasks are handled using OpenCV (cv2) and PIL, while Matplotlib and Seaborn are used for data visualization. NumPy and Pandas facilitate numerical operations and structured data handling.

Model evaluation is done with scikit-learn, using functions such as classification_report, average_precision_score, precision_recall_curve, f1_score, and compute_class_weight to assess model performance and manage imbalanced data.

## Additional Reference

Huggingface: https://huggingface.co/spaces/farrashv8/Insurance-fraud-claim-prediction
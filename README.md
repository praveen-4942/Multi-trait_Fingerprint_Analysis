

# 🧬 Multi-Trait Human Profiling from Fingerprints using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red?style=for-the-badge&logo=streamlit)

---

# 📖 Overview

This project presents a **non-invasive deep learning framework** for predicting multiple biological traits—**Blood Group, Gender, and Finger Type**—using fingerprint images. Traditional biological trait identification often requires invasive laboratory procedures or manual examination. This system eliminates the need for physical samples by leveraging **two custom-designed 22-layer Convolutional Neural Networks (CNNs)** that automatically learn fingerprint ridge patterns and minutiae-level features directly from fingerprint images.

The trained models are integrated into a **Streamlit web application**, enabling real-time prediction from a single uploaded fingerprint image for research and educational purposes.

---

# 🎯 Objectives

- Develop a deep learning model for fingerprint-based human profiling.
- Predict **Blood Group**, **Gender**, and **Finger Type** from fingerprint images.
- Build a non-invasive and automated biometric analysis system.
- Deploy the trained models through a real-time web application.

---

# ✨ Features

- 🩸 Blood Group Prediction (8 Classes)
- 👤 Gender Prediction
- ✋ Finger Type Classification (20 Classes)
- 🧠 Custom 22-Layer CNN Architecture
- ⚡ Real-Time Streamlit Deployment
- 🔍 Automatic Fingerprint Feature Learning
- 📊 Confusion Matrix & Performance Evaluation

---

# 📊 Performance

## Blood Group Prediction Model

| Metric | Value |
|---------|-------|
| Training Accuracy | **97.82%** |
| Validation Accuracy | **91.00%** |
| Test Accuracy | **91.06%** |

---

## Gender & Finger-Type Prediction Model

| Metric | Value |
|---------|-------|
| Training Accuracy | **80.44%** |
| Validation Accuracy | **91.24%** |
| Test Accuracy | **93.74%** |

---

# 📂 Datasets

This project utilizes two publicly available fingerprint datasets from Kaggle.

---

## 1️⃣ SOCOFing Dataset (Gender & Finger-Type)

**🔗 Dataset:** https://www.kaggle.com/datasets/ruizgara/socofing

### Description

The **SOCOFing (Sokoto Coventry Fingerprint Dataset)** contains approximately **55,000 fingerprint images** collected from **600 individuals**.

The dataset includes:

- Real fingerprint images
- Altered fingerprint images
  - Altered-Easy
  - Altered-Medium
  - Altered-Hard

Each fingerprint contains metadata including:

- Subject ID
- Gender
- Left / Right Hand
- Finger Type

The dataset was balanced into **20 classes** representing combinations of:

- Male / Female
- Left / Right Hand
- Thumb
- Index
- Middle
- Ring
- Little

This dataset was used for **Gender and Finger-Type Prediction**.

---

## 2️⃣ Fingerprint Dataset for Blood Group Classification

**🔗 Dataset:** https://www.kaggle.com/datasets/rohitpravinlohar/fingerprint-dataset-for-blood-group-classification

### Description

This dataset contains **8,000 fingerprint images** distributed equally across the eight ABO and Rh blood groups:

- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

Each blood group contains approximately **1,000 fingerprint images**, providing a balanced dataset for supervised learning.

This dataset was used to train the **Blood Group Prediction CNN**.

---

# 🧠 Model Architecture

The framework consists of **two independent custom 22-layer Convolutional Neural Networks (CNNs)**.

## 🔹 Model 1 – Blood Group Prediction

- Input Size: **256 × 256**
- Classes: **8**
- Optimizer: Adam
- Learning Rate: **0.001**
- Loss Function: Categorical Crossentropy

---

## 🔹 Model 2 – Gender & Finger-Type Prediction

- Input Size: **256 × 256**
- Classes: **20**
- Optimizer: Adam
- Learning Rate: **0.0001**
- Loss Function: Categorical Crossentropy

---

### CNN Components

Each model consists of:

- Convolution Layers
- Batch Normalization
- ReLU Activation
- MaxPooling Layers
- Dropout Layers
- Flatten Layer
- Dense Layers
- Softmax Output Layer

---

# ⚙️ Methodology

## 1. Data Collection

- Blood Group Fingerprint Dataset
- SOCOFing Dataset

---

## 2. Image Preprocessing

- Contrast Enhancement
- Brightness Adjustment
- Image Normalization
- Image Resizing (256 × 256)
- Dataset Balancing

---

## 3. Automatic Feature Learning

The CNN automatically learns hierarchical fingerprint features, including:

### Global Features

- Loops
- Whorls
- Arches

### Minutiae Features

- Ridge Endings
- Ridge Bifurcations
- Ridge Structures
- Core Points

---

## 4. Model Training

Both CNN models were trained using:

- Adam Optimizer
- Batch Normalization
- Dropout Regularization
- Early Stopping
- Model Checkpointing

---

## 5. Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 6. Deployment

The trained models are integrated into a **Streamlit web application** that enables users to upload a fingerprint image and instantly obtain predictions for:

- Blood Group
- Gender
- Finger Type

along with confidence scores.

---

# 🏗 System Workflow

```text
Fingerprint Image
        │
        ▼
Image Preprocessing
        │
        ├──────────────┐
        ▼              ▼
Blood Group CNN    Gender & Finger CNN
        │              │
        └──────┬───────┘
               ▼
      Streamlit Web Application
               ▼
        Final Predictions
```

---

# 💻 Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

---

# 📁 Project Structure

```text
project/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── blood_group_model.tflite
│   └── gender_finger_model.tflite
│
├── dataset/
│   ├── BloodGroup/
│   └── SOCOFing/
│
├── notebooks/
│
└── assets/
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

# 📌 Applications

- Forensic Investigation
- Emergency Healthcare
- Biometric Authentication
- Identity Verification
- Human Profiling Research
- Smart Security Systems

---

# 🔮 Future Scope

- Clinical validation using larger datasets
- Edge-device deployment
- Multi-modal biometric systems
- Federated Learning
- Explainable AI (XAI)
- Cloud deployment

---

# 👨‍💻 Authors

- **G. Praveenkumar**
- **V. Mounidharan**
- **P. J. Purushothaman**

### Faculty Mentor

**Dr. A. Divya**  
Assistant Professor (Sr. Gr.)  
Department of Electronics Engineering  
Madras Institute of Technology (MIT Campus)  
Anna University, Chennai

---

# 🙏 Acknowledgements

The authors sincerely thank the **Department of Electronics Engineering, Madras Institute of Technology, Anna University**, **Dr. A. Divya**, and the contributors of the publicly available Kaggle datasets for their guidance and support throughout this work.

---

# 📜 License

This project has been developed for **academic and research purposes only**.

The datasets remain the property of their respective authors and are subject to their respective Kaggle licenses and terms of use.

---

⭐ **If you found this project interesting, consider giving it a Star!**

# 🧬 Multi-Trait Human Profiling from Fingerprints using Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-Academic-lightgrey?style=for-the-badge)

---

# 📖 Overview

Traditional biological trait identification such as **blood group determination** requires invasive laboratory procedures. This project presents a **non-invasive deep learning framework** capable of simultaneously predicting:

- 🩸 Blood Group
- 👤 Gender
- ✋ Finger Type

using only a fingerprint image.

The system employs **two custom-designed 22-layer Convolutional Neural Networks (CNNs)** that automatically learn fingerprint ridge patterns and minutiae without handcrafted feature engineering.

A user-friendly **Streamlit web application** allows users to upload a fingerprint and receive instant predictions in real time.

---

# 🎯 Project Highlights

- ✅ Blood Group Prediction from Fingerprints
- ✅ Gender Prediction
- ✅ Finger Type Identification
- ✅ Custom 22-Layer CNN Architecture
- ✅ Real-Time Streamlit Deployment
- ✅ End-to-End Automated Pipeline
- ✅ No Manual Feature Extraction Required

---

# 📊 Results

| Task | Classes | Accuracy |
|------|---------|----------|
| Blood Group Prediction | 8 | **97.8%** |
| Gender + Finger Type Prediction | 20 | **96.7%** |

---

# 📂 Datasets

Two publicly available fingerprint datasets from Kaggle were used.

---

## 1️⃣ SOCOFing Dataset

**🔗 Link**

https://www.kaggle.com/datasets/ruizgara/socofing

### Description

SOCOFing (Sokoto Coventry Fingerprint Dataset) is one of the most widely used fingerprint datasets for biometric research.

It contains fingerprint images collected from **600 individuals**, including fingerprints from all ten fingers.

The dataset consists of:

- 6,000 Real fingerprint images
- 49,273 Altered fingerprint images
  - Altered-Easy
  - Altered-Medium
  - Altered-Hard

For this project, **only the Real fingerprint images** were used.

Each filename provides:

- Subject ID
- Gender
- Hand
- Finger Type

These labels were used to train a **20-class CNN** for simultaneous prediction of:

- Gender
- Finger Type

### Dataset Structure

```text
SOCOFing/
├── Real/
└── Altered/
    ├── Altered-Easy/
    ├── Altered-Medium/
    └── Altered-Hard/
```

---

## 2️⃣ Fingerprint Dataset for Blood Group Classification

**🔗 Link**

https://www.kaggle.com/datasets/rohitpravinlohar/fingerprint-dataset-for-blood-group-classification

### Description

This dataset contains fingerprint images labeled according to their respective human blood groups.

The dataset includes all eight major blood groups:

- A+
- A-
- B+
- B-
- AB+
- AB-
- O+
- O-

The dataset was used to train the **Blood Group Classification CNN**.

---

# 📌 Dataset Summary

| Dataset | Purpose | Classes |
|----------|----------|---------|
| SOCOFing | Gender + Finger Type Prediction | 20 |
| Blood Group Fingerprint Dataset | Blood Group Prediction | 8 |

---

# 🧠 Model Architecture

The project consists of **two independent custom CNN models**, each containing **22 layers**.

## 🔹 Model 1 — Blood Group Prediction

- Input Size: **256 × 256**
- Classes: **8**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Output: Blood Group

---

## 🔹 Model 2 — Gender & Finger Type Prediction

- Input Size: **256 × 256**
- Classes: **20**
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Output:
  - Gender
  - Finger Type

---

## CNN Components

Each CNN includes:

- Convolution Layers
- Batch Normalization
- ReLU Activation
- MaxPooling Layers
- Dropout Layers
- Fully Connected Layers
- Softmax Output Layer

---

# ⚙️ Methodology

## 1. Data Collection

- SOCOFing Dataset
- Blood Group Fingerprint Dataset

---

## 2. Image Preprocessing

- Contrast Enhancement
- Image Resizing (256 × 256)
- Normalization
- Dataset Balancing
- Train/Validation/Test Split

---

## 3. Model Training

- Custom 22-Layer CNN
- Adam Optimizer
- Batch Normalization
- Dropout Regularization

---

## 4. Model Evaluation

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## 5. Deployment

The trained models are deployed using **Streamlit**.

Workflow:

```
Fingerprint Image
        │
        ▼
Image Preprocessing
        │
        ▼
Blood Group CNN
        │
        ▼
Gender + Finger CNN
        │
        ▼
Prediction Results
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
Fingerprint-Human-Profiling/
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
│   ├── SOCOFing/
│   └── BloodGroup/
│
├── notebooks/
│
├── utils/
│
└── assets/
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Fingerprint-Human-Profiling.git
```

```bash
cd Fingerprint-Human-Profiling
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Application

```bash
streamlit run app.py
```

---

# 📈 Model Optimization

To reduce repository size and improve deployment performance:

- TensorFlow Lite Conversion
- Float16 Quantization

Benefits:

- Smaller model size (<25 MB)
- Faster inference
- Lower memory usage
- Suitable for edge deployment

---

# 📌 Applications

- Digital Forensics
- Criminal Investigation
- Biometric Authentication
- Medical Assistance
- Identity Verification
- Smart Healthcare
- Security Systems

---

# 🔮 Future Scope

- Clinical validation using larger datasets
- Mobile deployment
- Edge AI implementation
- Federated Learning
- Explainable AI (XAI)
- Multi-modal biometric fusion
- Cloud deployment using Docker & Kubernetes

---

# 👨‍💻 Authors

**G. Praveenkumar**

**V. Mounidharan**

**P. J. Purushothaman**

### Faculty Mentor

**Dr. A. Divya**

Department of Electronics Engineering

MIT Campus

Anna University, Chennai

---

# 🙏 Acknowledgements

We sincerely thank:

- Anna University, MIT Campus
- Department of Electronics Engineering
- Kaggle Dataset Contributors
- TensorFlow & Streamlit Communities

---

# 📜 License

This project has been developed **solely for academic and research purposes**.

The datasets used remain the property of their respective authors and are subject to the licenses and terms provided on Kaggle.

---

⭐ **If you found this project helpful, consider giving it a Star!**

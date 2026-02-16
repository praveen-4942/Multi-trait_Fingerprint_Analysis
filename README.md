# 🧬 Multi-Trait Human Profiling from Fingerprints using Deep Learning

## 📌 Overview
This project presents a non-invasive deep learning framework for simultaneous prediction of **Blood Group, Gender, and Finger Type** using fingerprint images. Traditional biological trait identification requires laboratory testing and physical sample collection. This system eliminates invasive procedures by leveraging custom 22-layer Convolutional Neural Networks (CNNs) to automatically extract complex ridge patterns and minutiae features directly from fingerprint images.

The trained models are deployed through a real-time Streamlit web application, enabling instant multi-trait prediction from a single fingerprint input.

---

## 🎯 Key Results
- ✅ 97.8% Accuracy – Blood Group Classification (8 Classes)  
- ✅ 96.7% Accuracy – Gender & Finger-Type Prediction (20 Classes)  
- ✅ Real-time Web Deployment  
- ✅ Fully Automated Feature Extraction (No Handcrafted Features)  

---

## 🧠 Model Architecture
Two custom-designed 22-layer CNN models were developed:

### 🔹 Model 1 – Blood Group Prediction
- 8-class classification  
- Dataset: 8,000 fingerprint images  
- Optimizer: Adam  
- Loss Function: Categorical Cross-Entropy  

### 🔹 Model 2 – Gender & Finger-Type Prediction
- 20 combined classes  
- Dataset: 55,000 fingerprint images  
- Strong generalization across unseen data  

Both models include:
- Convolutional Layers  
- Batch Normalization  
- MaxPooling  
- Dropout Layers  
- Fully Connected Layers  

---

## 🛠 Methodology
1. **Data Collection**
   - Ink-stamp fingerprints  
   - Secugen Hamster Pro20 Scanner  

2. **Preprocessing**
   - Contrast enhancement  
   - Image resizing to 256×256  
   - Dataset balancing  

3. **Model Training**
   - Custom 22-layer CNN  
   - Adam optimizer  
   - Batch normalization & dropout for stability  

4. **Evaluation Metrics**
   - Accuracy  
   - F1-Score  
   - Confusion Matrix  

5. **Deployment**
   - Streamlit-based web interface  
   - Real-time fingerprint upload  
   - Instant prediction with confidence scores  

---

## 📊 Performance
The proposed custom CNN outperformed pre-trained architectures such as **VGG16** and **EfficientNet** for fingerprint-specific feature extraction tasks.

---

## 💻 Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Matplotlib  
- Streamlit  

---

## 🚀 How to Run

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Application
```bash
streamlit run app.py
```

---

## 📁 Model Optimization
To reduce repository size and ensure GitHub compatibility, the model is optimized using:
- TensorFlow Lite Conversion  
- Float16 Quantization  

This reduces the model size below 25MB while maintaining high accuracy.

---

## 📌 Applications
- Forensic investigations  
- Emergency medical assistance  
- Biometric security systems  
- Non-invasive biological trait analysis  

---

## 🔮 Future Scope
- Clinical validation on larger datasets  
- Edge-device deployment  
- Multi-modal biometric integration  
- Federated learning for privacy preservation  

---

## 👨‍🎓 Authors
- G. Praveenkumar  
- V. Mounidharan  
- P. J. Purushothaman  

**Faculty Mentor:** Dr. A. Divya  
Department of Electronics Engineering  
MIT Campus, Anna University, Chennai  

---

## 📜 License
This project is developed for academic and research purposes.

# 😊 Smile Detection using Computer Vision

This project detects smiles in real-time using classical feature extraction technique.  
It consists of three main stages:

1. **Preprocessing and Dataset Preparation**
2. **Data Augmentation**
3. **Feature Extraction and Model Training**
4. **Real-Time Smile Detection via Webcam**
---
## 🧠 Project Summary

The goal of this project is to build a **binary classifier** (Smiling / Not Smiling) using image data from the **GENKI-4K dataset** in files.  
The project explores **handcrafted features(HOG + LBP)**  for classification using **SVM**.

---

## 🧩 Part 1 — Preprocessing

All preprocessing scripts are included in `preprocessing.py`.

**Steps include:**
- Detecting faces in the images using the Haar Cascade classifier.
- Cropping facial regions.
- Resizing cropped images to a uniform size (`256×256`).

**Output:**  
Processed images stored in a `processed_images/` folder.

---

## 🌀 Part 2 — Data Augmentation

Script: `augment_images.py`

**Goal:**  
To artificially expand the dataset and improve model generalization.

**Applied Augmentations:**
- Horizontal flips  
- Random rotations  
- Brightness changes  
- Random noise

**Output:**  
Augmented images saved in `aug_images/` folder.

---

## ⚙️ Part 3 — Feature Extraction and Model Training

All feature extraction logic and train the model are contained in `feature_extraction.py`.

### 🔹 Handcrafted Features
- **LBP (Local Binary Pattern):** Captures local texture information from grayscale facial regions.
- **HOG (Histogram of Oriented Gradients):** Encodes edge orientation and gradient patterns.

The combined feature vector (HOG + LBP) is used to train an **SVM classifier**.

### 🧮 Training
The script loads features and labels, splits data into training and testing sets, standardizes features, and trains the SVM model.

**Output:**
- `svm_model.joblib` (trained model)
- `scaler.joblib` (fitted scaler)
- `training_results` (performance metrics)

---

## 📷 Part 4 — Real-Time Smile Detection (Webcam)

Run `Testing_webcam.py` for real-time smile detection.

**Pipeline:**
1. Captures video frames from webcam.
2. Detects faces using **OpenCV Haar Cascade**.
3. Extracts features (HOG+LBP).
4. Predicts using the trained model.
5. Draws bounding boxes and overlays “Smiling” on detected faces.

## 📦 Requirements

* Python 3.x
* OpenCV(cv2)
* scikit-image
* scikit-learn
* Matplotlib
* NumPy
* natsort
* joblib
* nbimporter

## 🧭 How to Use

### 1️⃣ Clone the repository
```bash
git clone https://github.com/ZahraSafaei1272/SmileDetection_handcrafted.git
cd SmileDetection_handcrafted
```
### 2️⃣ Install dependencies
```bash
pip install opencv-python scikit-image scikit-learn  matplotlib numpy joblib natsort nbimporter
```
### 3️⃣ Run preprocessing
```bash
python preprocessing.py
```
### 4️⃣ Run data augmentation
```bash
python augment_images.py
```
### 5️⃣ Extract feature & train the model
```bash
python feature_extraction.py
```
### 6️⃣ Test with webcam
```bash
python Testing_with_webcam.py
```

## 📊 Results
### Metric	Value
* Training Accuracy	0.92
* Testing Accuracy	0.83


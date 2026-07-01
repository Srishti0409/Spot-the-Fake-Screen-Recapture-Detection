# Spot the Fake Photo – Screen Recapture Detection

A lightweight computer vision and machine learning system that detects whether an input image is:

- **0 → Real Photograph**
- **1 → Photograph of a Digital Screen (Screen Recapture)**

The system is designed as a fast, lightweight, and explainable anti-spoofing solution using handcrafted image features and an XGBoost classifier.

---

# Features

- Detects real photos vs. screen recaptures
- Uses handcrafted computer vision features instead of deep learning
- Lightweight and fast (~23 ms per image)
- No GPU required
- Command-line prediction interface
- Streamlit web application for interactive testing
- Automatic technical report generation after training

---

# Project Structure

```
Spot-the-Fake/
│
├── dataset/
│   ├── real/
│   └── screen/
│
├── feature_extractor.py
├── train.py
├── predict.py
├── streamlit_app.py
├── screen_detector.pkl
├── requirements.txt
├── README.md
├── report.md
└── .gitignore
```

---

# Approach

Instead of training a deep neural network, this project extracts handcrafted features that capture the physical characteristics of screen recaptures.

The extracted features include:

- **FFT (Fast Fourier Transform)** to detect Moiré patterns
- **Reflection Analysis** for glass glare detection
- **Laplacian Variance** for sharpness estimation
- **Local Binary Patterns (LBP)** for texture analysis
- Statistical image descriptors
- Frequency-domain characteristics

These features are fed into an **XGBoost Classifier**, which predicts the probability that the image is a screen recapture.

This approach provides:

- Better interpretability
- Lower computational cost
- Faster inference
- Small model size
- Good generalization with limited training data

---

# Tech Stack

- Python 3
- OpenCV
- NumPy
- Scikit-image
- Scikit-learn
- XGBoost
- Joblib
- Streamlit

---

# Installation

## Clone the repository

```bash
git clone https://github.com/Srishti0409/Spot-the-Fake-Screen-Recapture-Detection.git
cd Spot-the-Fake-Screen-Recapture-Detection
```

---

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.\.venv\Scripts\Activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

Store the images as follows:

```
dataset/

    real/
        image1.jpg
        image2.jpg
        ...

    screen/
        image1.jpg
        image2.jpg
        ...
```

- **real/** → Genuine photographs
- **screen/** → Photos taken of laptop/mobile/monitor screens

---

# Training

Train the model using:

```bash
python train.py
```

Training automatically:

- Extracts handcrafted features
- Splits the dataset
- Trains an XGBoost classifier
- Evaluates performance
- Saves the trained model (`screen_detector.pkl`)
- Generates `report.md`

---

# Prediction

Run prediction on a single image:

```bash
python predict.py image.jpg
```

Example:

```bash
python predict.py pothole_under_1MB.jpg
```

Output:

```
0.01
```

The script outputs **only one floating-point value** between **0 and 1**.

| Output | Meaning |
|---------|---------|
| 0.00 | Real Photo |
| 1.00 | Screen Photo |

---

# Streamlit Demo

Launch the web application:

```bash
streamlit run streamlit_app.py
```

Then open:

```
http://localhost:8501
```

Upload an image to receive a prediction along with extracted feature analysis.

---

# Performance

Training Results:

| Metric | Value |
|--------|-------|
| Accuracy | **97.50%** |
| Precision | **100.00%** |
| Recall | **95.00%** |
| F1 Score | **97.44%** |

Average Feature Extraction Latency:

```
23.36 ms per image
```

---

# Cost

Since the model runs entirely on-device:

- Cost per image: **Approximately $0**
- Cloud infrastructure: **Not required**

---

# Why This Approach?

Compared to deep learning models, handcrafted feature extraction offers:

- Lower memory usage
- Faster inference
- Better explainability
- No GPU dependency
- Excellent performance on small datasets

This makes it well suited for lightweight anti-spoofing applications.

---

# Future Improvements

- Larger and more diverse training dataset
- CNN / Vision Transformer comparison
- Mobile deployment
- Real-time webcam inference
- Automatic screen-region localization
- Quantized model for edge devices

---

# Repository

GitHub Repository:

https://github.com/Srishti0409/Spot-the-Fake-Screen-Recapture-Detection

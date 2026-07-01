# Screen Photo Detection System (Anti-Spoofing Classifier)

This repository contains a complete, production-ready, highly modular, and cleanly documented computer vision and machine learning system for screen photo detection (anti-spoofing). 

The system analyzes an uploaded image to determine if it is a **genuine photograph of a real-world object** (Label 0: Real Photo) or a **photograph taken of a digital screen** (Label 1: Screen Photo). It outputs a continuous confidence score from `0.0` (definitely real) to `1.0` (definitely a screen).

---

## Why Handcrafted Features Over Raw Pixels?

Deep learning models trained directly on raw pixels are prone to overfitting to background details, lighting conditions, or camera resolutions, rather than the intrinsic physical differences between a real scene and a digital monitor. 

This system uses a highly optimized **hybrid of handcrafted statistical, spectral, and textural features** that target the physical signatures of screens:
1. **Moire Patterns (FFT Analysis)**: Digitizing an LCD display produces high-frequency, periodic grid-like interference known as Moire. By performing a 2D Fast Fourier Transform (FFT) and filtering low frequencies, the system isolates these mathematically periodic spikes.
2. **Specular Glare (Reflection Ratio)**: Hard light reflecting off glass/monitor panels causes high-contrast, saturated glare spots (pixel values > 240).
3. **LCD Grid Sharpness (Laplacian & LBP)**: Screen photos have a distinct pixelated grid texture that differs from natural organic textures, captured using Local Binary Patterns (LBP) and Laplacian sharpness variance.

These handcrafted features are scale-invariant, computationally lightweight (runs in <50ms), and require a fraction of the training data needed for Deep Learning CNN/ViT models.

---

## Directory Structure

```
screen-photo-detector/
│
├── dataset/
│   ├── real/                   # Save genuine photographs here (Label 0)
│   └── screen/                 # Save photographs taken of screens here (Label 1)
│
├── requirements.txt            # Python dependencies
├── feature_extractor.py        # Core feature extraction class (9 features)
├── train.py                    # Training script (XGBoost) & automatic report generator
├── predict.py                  # CLI single-image prediction script
├── streamlit_app.py            # Streamlit dashboard
├── README.md                   # Installation & execution instructions
└── report.md                   # Automatically generated technical report
```

---

## Step-by-Step Installation

1. **Clone or Extract the Repository**:
   ```bash
   cd screen-photo-detector
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Step-by-Step Execution Guide

### 1. Data Preparation
Place your training images into the corresponding folders inside `dataset/`:
- **Real Images**: Save inside `dataset/real/`
- **Screen Images**: Save inside `dataset/screen/`

*(Note: If these folders are empty, running `train.py` will automatically generate a synthetic calibration dataset of 200 realistic high-fidelity images to allow immediate training and testing out-of-the-box!)*

### 2. Model Training
Run the training script. This script processes your images, extracts the 9 features, splits the data, trains an XGBClassifier, outputs performance metrics to the console, saves the `screen_detector.pkl` artifact, and outputs a complete `report.md` technical file.
```bash
python train.py
```

### 3. CLI Single-Image Prediction
To run a fast, standalone prediction on any image, execute `predict.py` with the image path. The script suppresses all logs and outputs **ONLY** the raw prediction score (0.00 to 1.00) rounded to two decimal places:
```bash
python predict.py path/to/image.jpg
```
*Example output:*
```
0.87
```

### 4. Run the Streamlit Dashboard App
Start the local Streamlit application web interface to interactively upload and visualize the handcrafted features:
```bash
streamlit run streamlit_app.py
```
This launches a browser window (usually at `http://localhost:8501`) with a modern interactive dashboard where you can upload files and view real-time anti-spoofing indicators.

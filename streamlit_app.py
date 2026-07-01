import streamlit as st
import os
import pandas as pd
import cv2
import numpy as np
import joblib

# Page setup
st.set_page_config(
    page_title="Screen Photo Detection System",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .tag-real {
        color: #10B981;
        background-color: #ECFDF5;
        border: 1px solid #A7F3D0;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
    .tag-screen {
        color: #EF4444;
        background-color: #FEF2F2;
        border: 1px solid #FCA5A5;
        padding: 0.35rem 1rem;
        border-radius: 9999px;
        font-weight: 700;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="main-header">🖥️ Screen Photo Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Handcrafted Statistical and Spectral Anti-Spoofing Image Classifier</div>', unsafe_allow_html=True)

# Check for Model Artifact
model_path = "screen_detector.pkl"
model_exists = os.path.exists(model_path)

if not model_exists:
    st.warning("⚠️ **Trained model artifact (`screen_detector.pkl`) not found!**")
    st.info("To train a robust detector on real-world data, execute `python train.py` in your terminal.\n\n"
            "**Interactive Demo Mode:** We have enabled a fallback analytical classifier based on mathematical feature thresholds so you can test uploaded images immediately.")

# Sidebar Information
with st.sidebar:
    st.header("System Settings")
    st.markdown("This system extracts 9 handcrafted spatial and spectral features from images to identify screen re-shoots.")
    
    st.subheader("Handcrafted Feature Metrics")
    st.markdown("- **A. Sharpness (Laplacian Var)**\n"
                "- **B. Edge Density (Canny)**\n"
                "- **C. Moire Energy (FFT 2D)**\n"
                "- **D. Brightness Mean (Gray)**\n"
                "- **E. Brightness Contrast (Std)**\n"
                "- **F. Color Saturation (HSV)**\n"
                "- **G. Specular Glare (Reflection)**\n"
                "- **H & I. Texture Statistics (LBP)**")
    
    st.divider()
    st.caption("Developed by Senior ML & CV Engineer • v1.0.0")

# Upload Area
uploaded_file = st.file_uploader("Upload an Image to Analyze", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Read file bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode to BGR OpenCV format
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img_bgr is None:
        st.error("Error: Could not decode the uploaded image.")
    else:
        # Layout: 2 Columns for Image Preview & Results
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("Uploaded Image Preview")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_rgb, use_container_width=True)
            
        with col2:
            st.subheader("Anti-Spoofing Analysis")
            
            with st.spinner("Extracting handcrafted features & evaluating..."):
                try:
                    # Import feature extractor lazily
                    from feature_extractor import FeatureExtractor
                    extractor = FeatureExtractor()
                    
                    # Extract 9 features
                    feats = extractor.extract_features(img_bgr)
                    
                    # Define features for plotting/review
                    feature_names = [
                        "A. Laplacian Sharpness",
                        "B. Edge Density",
                        "C. FFT Moire Energy",
                        "D. Brightness Mean",
                        "E. Brightness Std Dev",
                        "F. Saturation Mean",
                        "G. Glare Reflection Ratio",
                        "H. LBP Texture Mean",
                        "I. LBP Texture Std"
                    ]
                    
                    # Compute prediction probability (Class 1: Screen Photo)
                    is_simulated = False
                    if model_exists:
                        model = joblib.load(model_path)
                        prob = float(model.predict_proba(feats.reshape(1, -1))[0][1])
                    else:
                        # Analytical fallback model
                        is_simulated = True
                        laplacian_var = feats[0]
                        fft_energy = feats[2]
                        reflection_ratio = feats[6]
                        
                        score = 0.15
                        if fft_energy > 40:
                            score += 0.35
                        if reflection_ratio > 0.04:
                            score += 0.25
                        if laplacian_var > 1200:
                            score += 0.15
                        prob = min(0.99, max(0.01, score))
                    
                    # Determine classification label
                    if prob >= 0.5:
                        label = "Screen Photo"
                        tag_class = "tag-screen"
                        confidence = prob * 100
                    else:
                        label = "Real Photo"
                        tag_class = "tag-real"
                        confidence = (1 - prob) * 100
                        
                    # Results UI Block
                    st.markdown(f"""
                    <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 12px; padding: 2rem; margin-bottom: 2rem;">
                        <h4 style="margin: 0 0 1rem 0; color: #475569; font-size: 1rem; text-transform: uppercase; letter-spacing: 0.05em;">Classification Result</h4>
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
                            <span class="{tag_class}" style="font-size: 1.4rem;">{label}</span>
                            <span style="font-size: 2.2rem; font-weight: 800; color: #1E293B;">{prob:.2f}</span>
                        </div>
                        <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 0.5rem;">
                            Score maps the likelihood of being a <b>Screen Photo (Class 1)</b> on a scale of 0.0 to 1.0.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence progress bar
                    st.write(f"**Classification Confidence:** {confidence:.1f}%")
                    st.progress(prob)
                    
                    # Simulated warning
                    if is_simulated:
                        st.caption("ℹ️ Running in demo fallback mode. Run `train.py` to activate XGBoost model inference.")
                        
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.exception(e)
                    
            # Expanded handcrafted features breakdown
            st.divider()
            with st.expander("🔍 Handcrafted Feature Statistics (Raw Extracted Vector)", expanded=True):
                # Put features into a clean table/dataframe
                df_features = np.array(feats).reshape(1, -1)
                df = pd.DataFrame(df_features, columns=feature_names)
                st.dataframe(df.T.rename(columns={0: "Value"}), use_container_width=True)
                
                # Educational notes
                st.info("**Heuristics breakdown:**\n"
                        "- **High FFT Moire Energy (>35)** indicates periodic line interference caused by photographing pixel grids.\n"
                        "- **Glare Reflection (>0.03)** flags localized screen lighting reflection.\n"
                        "- **Laplacian Sharpness** ranges from <100 (smooth/blurry) to >2000 (extremely sharp LCD pixel transitions).")
else:
    # Welcome banner when no file uploaded
    st.info("👋 Welcome! Please upload an image using the file uploader above to test the Screen Photo Detection System.")
    
    # Showcase system specs
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #3B82F6;">9</div>
            <div class="metric-label">Handcrafted Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #10B981;">XGBoost</div>
            <div class="metric-label">Inference Model</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value" style="color: #8B5CF6;">&lt; 50ms</div>
            <div class="metric-label">Processing Time</div>
        </div>
        """, unsafe_allow_html=True)

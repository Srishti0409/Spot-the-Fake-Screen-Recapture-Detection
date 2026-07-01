import os
import cv2
import numpy as np
import warnings
from skimage.feature import local_binary_pattern

class FeatureExtractor:
    """
    FeatureExtractor extracts 9 handcrafted statistical and spectral features from an image
    to determine if it is a genuine photo of a real-world object (label 0)
    or a photo taken of a screen (label 1).
    """
    
    def __init__(self, lbp_radius=1, lbp_method='uniform'):
        self.lbp_radius = lbp_radius
        self.lbp_n_points = 8 * lbp_radius
        self.lbp_method = lbp_method

    def extract_features(self, image_path_or_array):
        """
        Processes a single image (from path or pre-loaded numpy array) and returns 
        a 1D NumPy array containing 9 handcrafted features.
        
        Features extracted:
        0. Laplacian Variance: Global sharpness indicator.
        1. Edge Density: Ratio of Canny edge pixels to total pixels.
        2. FFT Frequency Energy: High-frequency energy in the Fourier Spectrum (isolating Moire patterns).
        3. Brightness Mean: Mean of the grayscale image.
        4. Brightness Standard Deviation: Standard deviation of the grayscale image.
        5. Saturation Mean: Mean of the Saturation channel in HSV.
        6. Reflection Ratio: Ratio of pixels with grayscale value > 240 (detects monitor glare).
        7. LBP Histogram Mean: Mean of the Local Binary Patterns normalized histogram.
        8. LBP Histogram Std Dev: Standard deviation of the Local Binary Patterns normalized histogram.
        
        Returns:
            np.ndarray: 1D array of size (9,) containing the extracted features.
        """
        try:
            # 1. Load image
            if isinstance(image_path_or_array, str):
                if not os.path.exists(image_path_or_array):
                    raise FileNotFoundError(f"Image path does not exist: {image_path_or_array}")
                # Read image using OpenCV
                img = cv2.imread(image_path_or_array)
                if img is None:
                    raise ValueError(f"Failed to read image or invalid image format: {image_path_or_array}")
            elif isinstance(image_path_or_array, np.ndarray):
                img = image_path_or_array.copy()
            else:
                raise TypeError("Input must be a file path (str) or a numpy image array (np.ndarray)")

            # Check dimensions
            if len(img.shape) < 2:
                raise ValueError("Input image has invalid dimensions.")

            # If grayscale, convert to BGR placeholder to maintain HSV/Color logic
            if len(img.shape) == 2:
                gray = img
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Feature A: Laplacian Variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Feature B: Edge Density (Canny Edge Detection)
            # Use Otsu's thresholding to dynamically find Canny thresholds or standard 100/200
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size

            # Feature C: FFT Frequency Energy
            # Compute 2D Discrete Fourier Transform (FFT)
            f_transform = np.fft.fft2(gray.astype(np.float32))
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
            
            # Mask out the center low-frequency components (DC offset and low frequencies)
            h, w = gray.shape
            cy, cx = h // 2, w // 2
            # Use 10% of the image size as the low-frequency exclusion radius (at least 15 pixels)
            r = max(15, int(min(h, w) * 0.05))
            
            mask = np.ones((h, w), dtype=bool)
            mask[cy - r : cy + r, cx - r : cx + r] = False
            high_freq_spectrum = magnitude_spectrum[mask]
            fft_energy = np.mean(high_freq_spectrum) if high_freq_spectrum.size > 0 else 0.0

            # Feature D: Brightness Mean
            brightness_mean = np.mean(gray)

            # Feature E: Brightness Standard Deviation
            brightness_std = np.std(gray)

            # Feature F: Saturation Mean
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation_mean = np.mean(hsv[:, :, 1])

            # Feature G: Reflection Ratio
            # Monitors tend to have bright saturated white specular reflections
            reflection_ratio = np.sum(gray > 240) / gray.size

            # Feature H & I: Local Binary Patterns (LBP) Texture Statistics
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method=self.lbp_method)
            
            # Compute normalized histogram
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
            
            lbp_mean = np.mean(hist) if hist.size > 0 else 0.0
            lbp_std = np.std(hist) if hist.size > 0 else 0.0

            # Combine all 9 features into a 1D numpy array
            features = np.array([
                laplacian_var,       # A
                edge_density,        # B
                fft_energy,          # C
                brightness_mean,     # D
                brightness_std,      # E
                saturation_mean,     # F
                reflection_ratio,    # G
                lbp_mean,            # H
                lbp_std              # I
            ], dtype=np.float32)

            return features

        except Exception as e:
            # Return a default zero array or re-raise with detailed explanation
            raise RuntimeError(f"Error extracting features from image: {str(e)}")

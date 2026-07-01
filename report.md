# Screen Photo Detection System: Technical Performance Report

## 1. Project Objective
The goal of this project is to build an anti-spoofing computer vision pipeline that determines whether an uploaded image is a genuine photograph of a real-world object (**Label 0: Real Photo**) or a secondary reproduction taken of a digital display/monitor (**Label 1: Screen Photo**). This is vital for biometric authentication security, document verification, and digital fraud prevention.

## 2. Dataset Distribution Details
- **Total Samples processed**: 200
- **Class 0 (Real Photos)**: 100 (50.0%)
- **Class 1 (Screen Photos)**: 100 (50.0%)
- **Evaluation Split**: 80% Train, 20% Test (Stratified)

## 3. Comprehensive Feature Definitions
The system leverages **9 hand-crafted statistical, spectral, and textural features** extracted from images to ensure robust classification independent of deep-learning resource requirements:

1. **Laplacian Variance (Feature A)**: Measures global sharpness and high-frequency edge transition rates. Real photos exhibit a smooth continuous gradient, while photos of screens often have artificial high sharpness due to LCD pixel grids or blur from refocusing.
2. **Edge Density (Feature B)**: Ratio of Canny edge pixels to total image pixels. Screens display high-frequency scan lines and Moire textures that trigger excessive edge boundaries.
3. **FFT Frequency Energy (Feature C)**: Computed using a 2D Discrete Fourier Transform. The center low-frequency components are masked out to compute the mean energy of high frequencies. This isolates and detects periodic Moire grid patterns that are mathematically periodic.
4. **Brightness Mean (Feature D)**: The average grayscale intensity of the image.
5. **Brightness Standard Deviation (Feature E)**: Measures global contrast variations across the pixel spectrum.
6. **Saturation Mean (Feature F)**: The average saturation channel in the HSV color space. Screens often display highly saturated primary colors (RGB emissive channels) compared to natural reflections.
7. **Reflection Ratio (Feature G)**: Percentage of pixels exceeding a grayscale value of 240. This isolates and quantifies specular glare spots reflecting off shiny glass monitors.
8. **LBP Histogram Mean (Feature H)**: Local Binary Patterns (LBP) characterize fine-grain micro-textures. This represents the average bin density of uniform texture transitions.
9. **LBP Histogram Standard Deviation (Feature I)**: The standard deviation of the normalized LBP histogram, mapping the texture homogeneity across the image.

## 4. Performance Metrics
Based on the stratified test partition, the trained **XGBoost Classifier** achieved the following results:

- **Accuracy**: 0.9750 (97.50%)
- **Precision**: 1.0000 (100.00%)
- **Recall**: 0.9500 (95.00%)
- **F1 Score**: 0.9744 (97.44%)

### Confusion Matrix
```
               Predicted Real    Predicted Screen
Actual Real         20              0              
Actual Screen       1               19             
```

## 5. Execution Latency
- **Average Inference Latency**: 23.355 ms per image
  *(Includes file I/O, grayscale/HSV conversions, Canny edge detection, 2D FFT, LBP texture calculation, and XGBoost inference).*

## 6. Engineering Challenges
1. **Environmental Glare Variation**: Reflected light on glossy screens varies drastically based on ambient light sources, complicating glare ratio thresholding.
2. **Moire Variation Across Resolutions**: Periodic Moire lines appear differently depending on camera resolution, distance to screen, and display pixel pitch, requiring FFT frequency masks to be scale-invariant.
3. **Natural High-Contrast Textures**: Fine textures in real photos (e.g., woven fabrics, window blinds) can occasionally mimic Moire patterns, leading to potential false positives.

## 7. Architectural Steps for Future Improvement
1. **Scale-Invariant FFT Zooming**: Implement a pyramidal FFT search or normalize frequencies by camera intrinsic fields.
2. **Multi-Band Saturation Profiling**: Analyze saturation across different lighting zones to isolate emissive display backlighting.
3. **Deep Representation Merging**: Merge these lightweight handcrafted statistical features with a small, specialized CNN or MobileNet backbone to leverage both spatial and structural patterns.

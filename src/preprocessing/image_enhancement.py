# preprocessing/image_enhancement.py

import cv2
import numpy as np

def normalize_image(image):
    """
    Normalize image brightness and contrast
    """
    # Convert to grayscale if not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def reduce_noise(image):
    """
    Reduce noise while preserving edges
    """
    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(image, None, h=10, searchWindowSize=21, templateWindowSize=7)
    return denoised

def enhance_prescription(image_path):
    """
    Main function to enhance prescription image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Normalize
    normalized = normalize_image(img)
    
    # Denoise
    denoised = reduce_noise(normalized)
    
    # Binarize using adaptive thresholding
    binary = cv2.adaptiveThreshold(
        denoised, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Morphological operations to remove small noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def segment_regions(image):
    """
    Attempt to segment different regions of the prescription
    """
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by size
    significant_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Minimum area threshold
            significant_contours.append(contour)
    
    # Create regions
    regions = []
    for contour in significant_contours:
        x, y, w, h = cv2.boundingRect(contour)
        region = image[y:y+h, x:x+w]
        regions.append({
            'region': region,
            'position': (x, y, w, h)
        })
    
    return regions
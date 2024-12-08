#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:42:02 2024

@author: pingo
"""

# Steganalysis Example: Detecting Hidden Messages in Images

"""
Steps:
1. Prepare a dataset of stego images.
2. Apply statistical tests to detect embedded messages.
3. Differentiate between normal and stego images.
4. Document findings.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.stats import chisquare

# Part 1: Function to Embed a Message Using LSB Manipulation
def embed_message(input_image_path, hidden_msg, output_image_path="stego_image.png", interval=10):
    """Embed a hidden message into an image by manipulating the Least Significant Bit (LSB)."""
    
    #loading an image
    img = Image.open(input_image_path)
    img_array = np.array(img)           #converting to an array
    
    #here I am converting message to ascii value 
    ascii_values = [ord(char) for char in hidden_msg]

    img_flat = img_array.flatten()
    
    #here checling if image pixels are enough to encode a message
    if len(ascii_values) * interval > len(img_flat):
        print("Error: Image does not have enough pixels to encode the message.")
        return
    
    pixel_indices = [i * interval for i in range(len(ascii_values))]
    
    # Here emgbedding message into the image
    for index, char_value in zip(pixel_indices, ascii_values):
        img_flat[index] = char_value  
    
    #reshaped back to original image dimension
    modified_img = img_flat.reshape(img_array.shape).astype(np.uint8)
    
    
    #image saved
    Image.fromarray(modified_img).save(output_image_path)
    print(f"Stego image saved as '{output_image_path}'")



embed_message("doggo.png", "hidden message", "stego_doggo.png")

# Part 2: Statistical Analysis (Chi-Square Test) - here I an checking if stego image can be identified as "stego" by this test based on p-value

def analyze_images(original_image_path, modified_image_path):
    """Compare two images using histogram analysis and Chi-Square test."""
    
    # Load images in grayscale
    original_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    stego_img = cv2.imread(modified_image_path, cv2.IMREAD_GRAYSCALE)
    
    #Compute pixel intensity histograms
    orig_hist = cv2.calcHist([original_img], [0], None, [256], [0, 256]).flatten()
    stego_hist = cv2.calcHist([stego_img], [0], None, [256], [0, 256]).flatten()
    
    #Normalize histograms
    orig_hist = orig_hist / orig_hist.sum()
    stego_hist = stego_hist / stego_hist.sum()
    
    #Performing Chi-Square test
    chi_stat, p_val = chisquare(f_obs=stego_hist, f_exp=orig_hist)
    
    # Output results
    print(f"Chi-Square Statistic: {chi_stat}")
    print(f"P-Value: {p_val}")
    
    #Result analysis
    if p_val < 0.05:
        print("Significant difference detected: A hidden message may be present.")
    else:
        print("No significant difference detected: No evidence of a hidden message.")
    


# Example Usage: Analyze Images
analyze_images("doggo.png", "stego_doggo.png")





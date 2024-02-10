import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image, low_threshold, high_threshold):
    # Compute the minimum and maximum pixel values within the specified thresholds
    min_val = np.min(image)
    max_val = np.max(image)
    
    # Scale the pixel values to stretch the contrast
    stretched_image = (image - low_threshold) * (255.0 / (high_threshold - low_threshold))
    
    # Clip the result to ensure pixel values are within the valid range [0, 255]
    stretched_image = np.clip(stretched_image, 0, 255)
    
    return stretched_image

# Load the image
image = plt.imread('images/1.jpg')

# Convert to grayscale if necessary
if len(image.shape) > 2:
    image = np.mean(image, axis=2).astype(np.uint8)

# Define the low and high thresholds for contrast stretching
low_threshold = 50
high_threshold = 200

# Perform contrast stretching
stretched_image = contrast_stretching(image, low_threshold, high_threshold)

# Plot the original and stretched images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(stretched_image, cmap='gray')
plt.title('Contrast Stretched Image')
plt.axis('off')

plt.show()

# python contrast.py
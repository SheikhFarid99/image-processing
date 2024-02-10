import numpy as np
import matplotlib.pyplot as plt

def blend_images(image1, image2, alpha):
    # Ensure both images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape for blending"
    
    # Blend the images using the formula: output = alpha * image1 + (1 - alpha) * image2
    blended_image = alpha * image1 + (1 - alpha) * image2
    
    # Clip the result to ensure pixel values are within the valid range [0, 255]
    blended_image = np.clip(blended_image, 0, 255)
    
    return blended_image

# Load two images
image1 = plt.imread('images/1.jpg')
image2 = plt.imread('images/3.jpg')

# Convert to grayscale if necessary
if len(image1.shape) > 2:
    image1 = np.mean(image1, axis=2).astype(np.uint8)
if len(image2.shape) > 2:
    image2 = np.mean(image2, axis=2).astype(np.uint8)

# Define the blending parameter (alpha)
alpha = 0.5  # 0.5 means equal contribution from both images

# Perform image blending
blended_image = blend_images(image1, image2, alpha)

# Plot the original images and the blended image
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(blended_image, cmap='gray')
plt.title('Blended Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# python image_tran.py
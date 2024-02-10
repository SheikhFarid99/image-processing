import numpy as np
import matplotlib.pyplot as plt

def convolve(image, kernel):
    # Get the dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Initialize an empty output image
    output_image = np.zeros_like(image)
    
    # Pad the image to handle borders
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='edge')
    
    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            # Extract the region of interest
            roi = padded_image[i:i+kernel_height, j:j+kernel_width]
            
            # Apply the kernel
            output_image[i, j] = np.sum(roi * kernel)
    
    return output_image

# Load the image
image = plt.imread('images/1.jpg')

# Convert to grayscale if necessary
if len(image.shape) > 2:
    image = np.mean(image, axis=2).astype(np.uint8)

# Define the filter kernel (e.g., a simple averaging filter)
filter_kernel = np.array([[1, 1, 1],
                          [1, 1, 1],
                          [1, 1, 1]]) / 9  # Normalize to maintain brightness

# Perform convolution
filtered_image = convolve(image, filter_kernel)

# Plot the original and filtered images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Filtered Image')
plt.axis('off')

plt.show()

# python image_filtaring.py
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    # Initialize histogram with zeros
    histogram = [0] * 256

    # Iterate over each pixel in the image
    for row in image:
        for pixel in row:
            # Increment the corresponding histogram bin
            intensity = pixel
            histogram[intensity] += 1

    return histogram

def cumulative_distribution(histogram):
    # Compute cumulative distribution function
    cdf = [sum(histogram[:i+1]) for i in range(len(histogram))]

    # Normalize CDF
    cdf_normalized = [int((cdf_i - min(cdf)) / (np.prod(image.shape) - min(cdf)) * 255) for cdf_i in cdf]

    return cdf_normalized

def histogram_equalization(image, cdf_normalized):
    # Apply histogram equalization
    equalized_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            equalized_image[i, j] = cdf_normalized[image[i, j]]

    return equalized_image

# Load the image
image = plt.imread('./images/1.jpg')

# Convert to grayscale if necessary
if len(image.shape) > 2:
    image = np.mean(image, axis=2).astype(np.uint8)

# Compute histogram
histogram = calculate_histogram(image)

# Compute cumulative distribution function
cdf_normalized = cumulative_distribution(histogram)

# Perform histogram equalization
equalized_image = histogram_equalization(image, cdf_normalized)

# Plot original and equalized histograms
plt.figure(figsize=(10, 5))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(image.flatten(), bins=256, range=(0, 255), color='gray')
plt.title('Original Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.flatten(), bins=256, range=(0, 255), color='gray')
plt.title('Equalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# python histogram_equ.py
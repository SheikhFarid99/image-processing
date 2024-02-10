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
            intensity = int(intensity)
            histogram[intensity] += 1

    return histogram

def plot_histogram(histogram):
    # Plot histogram
    plt.plot(histogram, color='gray')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Histogram of Grayscale Image')
    plt.show()

# Load the image
image = plt.imread('./images/1.jpg')

# Convert to grayscale if necessary
if len(image.shape) > 2:
    image = np.mean(image, axis=2)

# Compute histogram
histogram = calculate_histogram(image)

# Plot histogram
plot_histogram(histogram)

# python histogram.py
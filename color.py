import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    # Initialize histograms for each color channel
    hist_R = np.zeros(256, dtype=int)
    hist_G = np.zeros(256, dtype=int)
    hist_B = np.zeros(256, dtype=int)

    # Iterate over each pixel in the image
    for row in image:
        for pixel in row:
            # Increment the corresponding histogram bin for each color channel
            hist_R[pixel[0]] += 1  # Red channel
            hist_G[pixel[1]] += 1  # Green channel
            hist_B[pixel[2]] += 1  # Blue channel

    return hist_R, hist_G, hist_B

# Load the image
image = plt.imread('images/1.jpg')

# Calculate histograms for each color channel
hist_R, hist_G, hist_B = calculate_histogram(image)

# Plot the histograms
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.bar(range(256), hist_R, color='red', alpha=0.7)
plt.title('Red Channel Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.bar(range(256), hist_G, color='green', alpha=0.7)
plt.title('Green Channel Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.bar(range(256), hist_B, color='blue', alpha=0.7)
plt.title('Blue Channel Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# python color.py

import numpy as np
import matplotlib.pyplot as plt

def draw_diagonal_cross(image):
    # Get the dimensions of the image
    modified_image = image.copy()
    height, width = image.shape[:2]

    # Draw diagonal lines
    for i in range(min(height, width)):
        modified_image[i, i] = [255, 0, 0]  # Blue color pixel on the main diagonal
        modified_image[i, width - i - 1] = [255, 0, 0]  # Blue color pixel on the opposite diagonal

# Load the image
image = plt.imread('images/1.jpg')

# Ensure the image has three channels (RGB)
if len(image.shape) < 3:
    # If the image is grayscale, convert it to RGB
    image = np.stack((image,) * 3, axis=-1)

# Draw the diagonal cross on the image
draw_diagonal_cross(image)

# Plot the image with the diagonal cross
plt.imshow(image)
plt.title('Image with Diagonal Cross')
plt.axis('off')
plt.show()

# python diagonal_any.py

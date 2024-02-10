import numpy as np
import matplotlib.pyplot as plt

def add_boundary(image, boundary_width, boundary_value):
    # Get the dimensions of the original image
    height, width = image.shape

    # Create a new array with extended dimensions
    extended_image = np.full((height + 2 * boundary_width, width + 2 * boundary_width), boundary_value, dtype=image.dtype)

    # Copy the original image into the center of the extended array
    extended_image[boundary_width:boundary_width + height, boundary_width:boundary_width + width] = image

    return extended_image

# Load the image
image = plt.imread('images/1.jpg')

# Convert to grayscale if necessary
if len(image.shape) > 2:
    image = np.mean(image, axis=2).astype(np.uint8)

# Define boundary width and value
boundary_width = 20
boundary_value = 0  # Black color for the boundary

# Add boundary to the image
image_with_boundary = add_boundary(image, boundary_width, boundary_value)

# Plot the image with boundary
plt.imshow(image_with_boundary, cmap='gray')
plt.title('Image with Boundary')
plt.axis('off')
plt.show()

# python image_boundary.py

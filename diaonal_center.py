import numpy as np
import matplotlib.pyplot as plt

def add_diagonal_boundary(image, boundary_value, center_value):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Create a new array with the same dimensions as the original image
    new_image = np.zeros_like(image)

    # Set the boundary pixels
    for i in range(height):
        for j in range(width):
            if i == j:
                new_image[i, j] = boundary_value
            elif i == width - j - 1:
                new_image[i, j] = boundary_value

    # Assign the center value
    new_image[height // 4: 3 * height // 4, width // 4: 3 * width // 4] = center_value

    return new_image

# Define the size of the square image
size = 200

# Create the initial image (black background)
image = np.zeros((size, size), dtype=np.uint8)

# Define boundary value (white) and center value (gray)
boundary_value = 255
center_value = 128

# Add diagonal boundary and assign center value
image_with_diagonal_boundary = add_diagonal_boundary(image, boundary_value, center_value)

# Plot the image
plt.imshow(image_with_diagonal_boundary, cmap='gray')
plt.title('Image with Diagonal Boundary and Center Value')
plt.axis('off')
plt.show()
# python diaonal_center.py
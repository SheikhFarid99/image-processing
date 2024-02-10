import numpy as np
import matplotlib.pyplot as plt

def create_diagonal_cross_square(size):
    # Initialize an array with all zeros
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Draw diagonal lines
    for i in range(size):
        image[i, i] = 255  # White pixel on the main diagonal
        image[i, size - i - 1] = 255  # White pixel on the opposite diagonal
    
    return image

# Define the size of the square image
size = 200

# Create the diagonal cross square image
image = create_diagonal_cross_square(size)

# Plot the image
plt.imshow(image, cmap='gray')
plt.title('Diagonal Cross Square Image')
plt.axis('off')
plt.show()

# python diagonal.py
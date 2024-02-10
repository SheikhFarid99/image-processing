import numpy as np
import matplotlib.pyplot as plt

def add_images(image1, image2):
    # Ensure both images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape for addition"
    
    # Add the pixel values of the images
    result_image = image1 + image2
    
    # Clip the result to ensure pixel values are within the valid range [0, 255]
    result_image = np.clip(result_image, 0, 255)
    
    return result_image

def subtract_images(image1, image2):
    # Ensure both images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape for subtraction"
    
    # Subtract the pixel values of the images
    result_image = image1 - image2
    
    # Clip the result to ensure pixel values are within the valid range [0, 255]
    result_image = np.clip(result_image, 0, 255)
    
    return result_image

def multiply_images(image1, image2):
    # Ensure both images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape for multiplication"
    
    # Multiply the pixel values of the images
    result_image = image1 * image2
    
    # Clip the result to ensure pixel values are within the valid range [0, 255]
    result_image = np.clip(result_image, 0, 255)
    
    return result_image

def divide_images(image1, image2):
    # Ensure both images have the same shape
    assert image1.shape == image2.shape, "Images must have the same shape for division"
    
    # Divide the pixel values of the images, handling division by zero by adding a small value
    result_image = np.divide(image1, image2, out=np.zeros_like(image1), where=image2!=0)
    
    # Clip the result to ensure pixel values are within the valid range [0, 255]
    result_image = np.clip(result_image, 0, 255)
    
    return result_image

# Load two images
image1 = plt.imread('images/1.jpg')
image2 = plt.imread('images/3.jpg')

# Convert to grayscale if necessary
if len(image1.shape) > 2:
    image1 = np.mean(image1, axis=2).astype(np.uint8)
if len(image2.shape) > 2:
    image2 = np.mean(image2, axis=2).astype(np.uint8)

# Perform addition, subtraction, multiplication, and division of images
add_result = add_images(image1, image2)
subtract_result = subtract_images(image1, image2)
multiply_result = multiply_images(image1, image2)
divide_result = divide_images(image1, image2)

# Plot the results
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(add_result, cmap='gray')
plt.title('Addition')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(subtract_result, cmap='gray')
plt.title('Subtraction')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(multiply_result, cmap='gray')
plt.title('Multiplication')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(divide_result, cmap='gray')
plt.title('Division')
plt.axis('off')

plt.tight_layout()
plt.show()

# python image_oparetion.py
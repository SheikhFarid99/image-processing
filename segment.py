import numpy as np
import cv2

def segment_image(image_path, num_clusters):
    # Read the image
    image = cv2.imread(image_path)
    
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    
    # Convert to float type
    pixel_values = np.float32(pixel_values)
    
    # Define criteria for K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Perform K-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert centers to integer type
    centers = np.uint8(centers)
    
    # Map each pixel to its corresponding center
    segmented_image = centers[labels.flatten()]
    
    # Reshape back to original image shape
    segmented_image = segmented_image.reshape(image.shape)
    
    return segmented_image

# Example usage
input_image = './images/1.jpg'  # Replace with your input image path
output_image_path = './images/segment.jpg'  # Specify output image path
num_clusters = 5  # Specify the number of clusters for segmentation

# Perform image segmentation
segmented_image = segment_image(input_image, num_clusters)

# Display and save the segmented image
cv2.imshow('Segmented Image', segmented_image)
cv2.imwrite(output_image_path, segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# python segment.py

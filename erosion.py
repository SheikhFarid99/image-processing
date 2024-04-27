import cv2
import matplotlib.pyplot as plt
import numpy as np

image= cv2.imread("./images/erosion.png",0)

height,width= image.shape 

#erosing....
# kernel = np.ones((15,15), dtype=np.uint8)

# constant= (15-1)//2

# imgErode= np.zeros((height,width), dtype=np.uint8)


# for i in range(constant, height-constant):
#   for j in range(constant,width-constant):
#     temp= image[i-constant:i+constant+1, j-constant:j+constant+1]
#     product= temp*kernel
#     imgErode[i,j]= np.min(product)

#dilation....

kernal = 15
SE = np.array([[0,1,0], [1,1,1],[0,1,0]])

constant= 1

imgErode= np.zeros((height,width), dtype=np.uint8)


for i in range(constant, height-constant):
  for j in range(constant,width-constant):
    temp= image[i-constant:i+constant+1, j-constant:j+constant+1]
    product= temp*SE
    imgErode[i,j]= np.max(product)

plt.subplot(1,2,1)
plt.imshow(image, cmap="gray")
plt.title('Orginal Image')

plt.subplot(1,2,2)
plt.imshow(imgErode, cmap="gray")
plt.title('Dilation Image')

plt.tight_layout()
plt.show()

# python erosion.py
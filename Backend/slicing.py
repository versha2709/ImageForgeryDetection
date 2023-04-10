import cv2
import numpy as np

# Load the image
img = cv2.imread('slice.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set the block size and search area size
block_size = 16
search_area = 64

# Compute the Laplacian of the image
laplacian = cv2.Laplacian(gray, cv2.CV_64F)

# Divide the image into blocks and compute the variance of the Laplacian for each block
var_laplacian = np.zeros_like(gray)
for i in range(0, gray.shape[0] - block_size, block_size):
    for j in range(0, gray.shape[1] - block_size, block_size):
        block = laplacian[i:i+block_size, j:j+block_size]
        var_laplacian[i:i+block_size, j:j+block_size] = np.var(block)

# Find the maximum variance of the Laplacian in each search area
max_var_laplacian = np.zeros_like(gray)
for i in range(0, gray.shape[0] - search_area, search_area):
    for j in range(0, gray.shape[1] - search_area, search_area):
        search_region = var_laplacian[i:i+search_area, j:j+search_area]
        max_index = np.argmax(search_region)
        max_i, max_j = np.unravel_index(max_index, search_region.shape)
        max_var_laplacian[i+max_i, j+max_j] = search_region[max_i, max_j]

# Threshold the maximum variance of the Laplacian to identify potential spliced regions
thresh = np.max(max_var_laplacian) * 0.8
potential_regions = np.where(max_var_laplacian >= thresh)

# Draw rectangles around the potential spliced regions
for i, j in zip(potential_regions[0], potential_regions[1]):
    cv2.rectangle(img, (j-block_size//2, i-block_size//2), (j+block_size//2, i+block_size//2), (0, 0, 255), 2)

# Display the image with potential spliced regions highlighted
cv2.imshow('Splicing Forgery Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Load the image
img = cv2.imread('compressed.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set the block size and search area size
block_size = 12
search_area = 64

# Compute the DCT of the image
dct = cv2.dct(np.float32(gray))

# Divide the image into blocks and compute the standard deviation of the DCT for each block
std_dct = np.zeros_like(gray)
for i in range(0, gray.shape[0] - block_size, block_size):
    for j in range(0, gray.shape[1] - block_size, block_size):
        block = dct[i:i+block_size, j:j+block_size]
        std_dct[i:i+block_size, j:j+block_size] = np.std(block)

# Find the maximum standard deviation of the DCT in each search area
max_std_dct = np.zeros_like(gray)
for i in range(0, gray.shape[0] - search_area, search_area):
    for j in range(0, gray.shape[1] - search_area, search_area):
        search_region = std_dct[i:i+search_area, j:j+search_area]
        max_index = np.argmax(search_region)
        max_i, max_j = np.unravel_index(max_index, search_region.shape)
        max_std_dct[i+max_i, j+max_j] = search_region[max_i, max_j]

# Threshold the maximum standard deviation of the DCT to identify potential compressed regions
thresh = np.max(max_std_dct) * 0.8
potential_regions = np.where(max_std_dct >= thresh)

# Draw rectangles around the potential compressed regions
for i, j in zip(potential_regions[0], potential_regions[1]):
    cv2.rectangle(img, (j-block_size//2, i-block_size//2), (j+block_size//2, i+block_size//2), (0, 0, 255), 2)

# Display the image with potential compressed regions highlighted
cv2.imshow('Compression Forgery Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

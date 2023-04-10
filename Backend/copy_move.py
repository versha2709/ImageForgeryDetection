import cv2
import numpy as np

# Load the image
img = cv2.imread('addy.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Set the block size and stride
block_size = 16
stride = 8

# Compute the difference image using the block matching algorithm
diff = cv2.absdiff(cv2.resize(gray, (gray.shape[1] // stride, gray.shape[0] // stride)),
                   cv2.resize(cv2.blur(gray, (block_size, block_size)), (gray.shape[1] // stride, gray.shape[0] // stride)))

# Threshold the difference image to identify potential copy-move regions
thresh = np.max(diff) * 0.8
potential_regions = np.where(diff >= thresh)

# Draw rectangles around the potential copy-move regions
for i, j in zip(potential_regions[0], potential_regions[1]):
    cv2.rectangle(img, (j*stride, i*stride), ((j+1)*stride, (i+1)*stride), (0, 0, 255), 2)

# Display the image with potential copy-move regions highlighted
cv2.imshow('Copy-Move Forgery Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

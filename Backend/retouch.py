import cv2
import numpy as np

# Load the image
img = cv2.imread('alle.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the Laplacian of the image
lap = cv2.Laplacian(gray, cv2.CV_64F)

# Compute the absolute value of the Laplacian and convert it to uint8
abs_lap = np.uint8(np.absolute(lap))

# Threshold the Laplacian to identify potential retouched regions
thresh = np.max(abs_lap) * 0.8
potential_regions = np.where(abs_lap >= thresh)

# Draw rectangles around the potential retouched regions
for i, j in zip(potential_regions[0], potential_regions[1]):
    cv2.rectangle(img, (j-10, i-10), (j+10, i+10), (0, 0, 255), 2)

# Display the image with potential retouched regions highlighted
cv2.imshow('Retouching Forgery Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

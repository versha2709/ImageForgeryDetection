import cv2
import cv2 as cv #python library image processing ke liye,bohot
import numpy as np # working with arrays;linear algebra

img1 = cv2.imread('original.jpg', 0) #function loads image from a specific file
img2 = cv2.imread('forged.png', 0)

img1 = cv2.resize(img1, (300, 200))
img2 = cv2.resize(img2, (300, 200))

# Initialize SURF detector
#surf = cv2.xfeatures2d.SIFT_create() #sift algorithm used for

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)


# Find keypoint and descriptors in both images
#kp1, des1 = surf.detectAndCompute(img1, None) #same function detects the keypoint and then makes the descriptors
#kp2, des2 = surf.detectAndCompute(img2, None)

#for d in des1: #to see that for every feature found we have an array of descriptors
 #   print(d)
# Initialize Brute Force Matcher
#bf = cv2.BFMatcher() #function to match descriptors by finding closest descriptor in first set to second set

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=2)

#img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched image


cv2.imshow('Matched Features', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate average distance of matches
distances = [match.distance for match in matches]
avg_distance = sum(distances)/len(distances)

# Print average distance
# print("Average Distance: ", avg_distance)




if avg_distance ==0 :
    print("not Forged")
else:
    print("Forged")
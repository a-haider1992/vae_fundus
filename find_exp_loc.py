import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import random as rng

# Read the big and small images
big_image = cv2.imread('image7.jpg')
small_image = cv2.imread('explanation-found-image7.jpg')

# Compute convex hull points of the small image
gray_small = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)

# Threshold parameters
threshold = 80
# Noise reduction parameter - larger value means more noise reduction - smaller contours are considered noise
min_contour_area = 1000

# Canny edge detection
canny_output = cv2.Canny(gray_small, threshold, threshold * 2)

# Find contours
contours, _ = cv2.findContours(canny_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the convex hull object for each contour
hull_list = []
for i in range(len(contours)):
    hull = cv2.convexHull(contours[i])
    hull_list.append(hull)

# Draw contours + hull results
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
    cv2.drawContours(drawing, contours, i, color)
    cv2.drawContours(drawing, hull_list, i, color)
    cv2.imwrite('contours.jpg', drawing)

# Calculate the scaling factors for x and y coordinates
x_scale_factor = big_image.shape[1] / small_image.shape[1]
y_scale_factor = big_image.shape[0] / small_image.shape[0]

print("x_scale_factor:", x_scale_factor)
print("y_scale_factor:", y_scale_factor)

# Scale the hull points
scaled_hull = np.copy(hull)
scaled_hull[:, 0, 0] = (scaled_hull[:, 0, 0] * x_scale_factor).astype(int)
scaled_hull[:, 0, 1] = (scaled_hull[:, 0, 1] * y_scale_factor).astype(int)

# Use convex hull points to find the location in the big image
x_min, y_min = np.min(scaled_hull[:, 0, :], axis=0)
x_max, y_max = np.max(scaled_hull[:, 0, :], axis=0)

# Create a mask with the same size as the big image
mask = np.zeros_like(big_image)

# Draw the convex hull points on the mask
cv2.drawContours(mask, [scaled_hull], 0, (255, 255, 255), -1)
cv2.imwrite('mask.jpg', mask)

# Apply the mask to the big image to create the masked big image
masked_big_image = cv2.bitwise_and(big_image, mask)

# Save the masked big image
# cv2.imwrite('masked_big_image.jpg', masked_big_image)

# Compute the size of the patch from the perspective of the resolution of the big image
patch_size = (x_max - x_min, y_max - y_min)

# Print the location of the region in the big image and the patch size
print("Location of region in big image: (x_min, y_min):", x_min, y_min, "(x_max, y_max):", x_max, y_max)
print("Patch size required:", patch_size)

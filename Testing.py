import os
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np

from Transformation_Functions import image_flipper, box_flipper, image_rotator, box_rotator, rect_draw

#--------------- TESTING ---------------#

# This is a  testing file for so you can test the Transformation_Functions functions.
# You can test the function on individual images so you can see how they work

# Path to image and label
path = r'C:/Users/david/Documents/ISI Placement/i3dr/Cracks/Labelled Dataset/images/20210719_100727_370_r_rect.png'
label= r'C:/Users/david/Documents/ISI Placement/i3dr/Cracks/Labelled Dataset/labels/20210719_100727_370_r_rect.txt'

# Reading in image and label
image = cv2.imread(path)
coords = np.loadtxt(label, delimiter=' ')

# Flips the image in all 3 axes
axis = [-1, 0, 1]
for ax in axis:
    flipped_image, w, h=image_flipper(image, ax)
    flipped_coords = box_flipper(coords, w, h, ax)

    # Display the flipped image and flipped coordinate box so you can see it
    image_show= rect_draw(flipped_image, flipped_coords, 6)
    plt.imshow(image_show)
    plt.show()

# Rotates the image by the defined angles
angles = [0, 15, 45, 90, 180, 270]
for a in angles:
    angle=int(a)
    rotated_image, width, height = image_rotator(image, angle )
    rotated_coords = box_rotator(coords,width, height, angle)

    # Display the rotated image and rotated coordinate box so you can see it
    image_show= rect_draw(rotated_image, rotated_coords, 6)
    plt.imshow(image_show)
    plt.show()
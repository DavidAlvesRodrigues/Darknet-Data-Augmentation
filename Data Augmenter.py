import os
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
from Transformation_Functions import image_flipper, box_flipper, image_rotator, box_rotator, rect_draw

# --------------- MASS PRODUCTION ---------------#
# You can run the code and mass produce augmented pictures.
# You need to do 3 things, define the path to the images and labels,
# define path to save images and labels, and define the angles you want to
# rotate your image by. Then run the code.

# Path to images, labels, and save folder.
image_folder = r"C:/Users/david/Documents/ISI Placement/i3dr/Cracks/Labelled Dataset/images"
label_folder = r"C:/Users/david/Documents/ISI Placement/i3dr/Cracks/Labelled Dataset/labels"
save_image = r"C:/Users/david/Documents/ISI Placement/i3dr/Cracks/testting/images"
save_label = r"C:/Users/david/Documents/ISI Placement/i3dr/Cracks/testting/labels"

# reading in image files
images = []
for filename in os.listdir(image_folder):
    image = cv2.imread(os.path.join(image_folder,filename))
    if image is not None:
        images.append(image)
        
# reading in label files
labels = []
for filename in os.listdir(label_folder):
    lab = np.loadtxt(os.path.join(label_folder,filename), delimiter=' ')
    if lab is not None:
        labels.append(lab)

# Rotating images
for i in range(0, len(images)):

    img = images[i]
    coords = labels[i]
    angles = [0, 15, 45, 90, 180, 270]

    for a in angles:
        angle=int(a)

        rotated_img, width, height = image_rotator(img, angle)
        rotated_coords = box_rotator(coords, width, height, angle)

        # saving the rotated image to a path
        path_image_flip = os.path.join(save_image,"manhole-"+str(i)+"-"+str(angle)+".jpg")
        cv2.imwrite(path_image_flip, rotated_img)

        # saving the rotated coordinates to a path
        path_label_flip = os.path.join(save_label,"manhole-"+str(i)+"-"+str(angle)+".txt")
        np.savetxt(path_label_flip ,rotated_coords, delimiter=' ')

        # displaying the rotated image and rotated coordinates. 
        image_show= rect_draw(rotated_img, rotated_coords, 6)
        plt.imshow(image_show)
        plt.show()

# Flipping Images
for i in range(0, len(images)):
    img = images[i]
    coords = labels[i]

    axis = [-1, 0, 1]
    for ax in axis:

        flipped_image, w, h=image_flipper(img, ax)
        flipped_coords = box_flipper(coords, w, h, ax)

        if ax == -1 :
            name ="-both"
        if ax == 0 :
            name ="-x"
        if ax == 1 :
            name ="-y"

        path_image = os.path.join(save_image,"manhole-"+str(i)+name+".jpg")
        cv2.imwrite(path_image, flipped_image)

        path_label = os.path.join(save_label,"manhole-"+str(i)+name+".txt")
        np.savetxt(path_label ,flipped_coords, delimiter=' ')

        # displaying the flipped image and flipped coordinates
        image_show= rect_draw(flipped_image, flipped_coords, 6)
        plt.imshow(image_show)
        plt.show()

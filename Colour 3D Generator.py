# # This code was written by David Alves Rodrigues,
# # with major help from Ben Knight, especially when it came
# # generating the point cloud.

# Importing libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import random
import os
from skimage import color, img_as_float, io, img_as_ubyte
from phase.core.types import StereoMatcherType, MatrixUInt8
from phase.core.stereomatcher import StereoI3DRSGM, StereoParams
from phase.core.calib import StereoCameraCalibration
from phase.core import normaliseDisparity, processStereo, savePLY, disparity2xyz


#--------------- FUNCTIONS ---------------#


def colorise(image, hue:float, saturation=1):
    """
    Add color of the given hue to an RGB image.
    Inputs:
    Image - numpy.array
    Hue   - a float. This is the colour you want
    Saturation - how saturated the colour is.

    Outputs:
    Coloured_Image - numpy.array of coloured image
    """

    hsv = color.rgb2hsv(image)
    hsv[:, :, 1] = saturation
    hsv[:, :, 0] = hue
    return color.hsv2rgb(hsv)

def get_coords(image, coords):
    """
    Unpacks the darknet label file for a specific image into an array
    which is then used to highlight the feature in 3d.
    Inputs:
    Image  - np.array (your image)
    Coords - (coordinates) is a np.array, this is your darknet label file

    Outputs:
    pts - Coordinates of the boundary box
    """
    height, width = image.shape[:2]
    coords = coords.reshape((-1,5))

    pts = np.empty((0,4), int)
    for i in range(0,np.shape(coords)[0]):
        w = coords[i,3]*width
        h = coords[i, 4]*height
        x = (coords[i,1]*width)-w/2
        y = (coords[i,2]*height)-h/2
        sides = np.array([[x, x + w, y, y + h]], dtype= np.int32)
        pts = np.append(pts, sides, axis=0)
        pts = pts.reshape((-1,4))
    return pts


#--------------- Giving hue to area inside boundary box of only left images ---------------#


# # Path to images, labels, and save folder.
images_folder = "Data\images"
label_folder = "Data\labels"
save_folder = "Data\save"

if not os.path.exists(images_folder): raise Exception('Invalid path to image folder')
if not os.path.exists(label_folder): raise Exception('Invalid path to label folder')
if not os.path.exists(save_folder): raise Exception('Invalid path to save folder')

# # loads 2D stereo left images from a folder into list of arrays, which can be iterated through.
images = []
basenames = []
for filename in glob.glob(images_folder +'\*_l_rect.png'):
    basename = os.path.basename(filename)
    img = cv2.imread(filename)
    if img is not None:
        images.append(img)
        basenames.append(basename)

# # Loads labels for 2D left images of manhole covers into list
labels = []
for filename in glob.glob(label_folder +'\*_l_rect.txt'):
    lab = np.loadtxt(filename, delimiter=' ')
    if lab is not None:
        labels.append(lab)

# # Take left image, and add hue to area inside the boundary box.
for i in range(0, len(images)):
    basename = basenames[i]
    image = images[i]
    grayscale_image = img_as_float(image)
    image = color.gray2rgb(grayscale_image)

    coords = labels[i]
    points = get_coords(image, coords)

    cracked_image = image.copy()

    # Iterate through every boundary box for a specific image,
    # adding hue to area inside boundary box
    for r in range(0, np.shape(points)[0]):
        boundary_box = (slice(points[r, 2],points[r, 3]), slice(points[r, 0],points[r, 1]))
        hue = random.uniform(0, 1) # adds a random colour for each bounday box
        cracked_image[boundary_box] = colorise(image[boundary_box], hue, saturation=0.6)

    # # Display images for you to see.
    # fig, ax1= plt.subplots(figsize=(8, 4))
    # ax1.imshow(cracked_image)
    # plt.show()

    # # Saves image to the following folder
    cracked_image = img_as_ubyte(cracked_image)
    path = os.path.join(save_folder,basename)
    io.imsave(path, cracked_image)


#--------------- Generating 3D for all images in a folder ---------------#


# # Checking if phase license is valid
license_valid = StereoI3DRSGM().isLicenseValid()
if license_valid:
    print("I3DRSGM license accepted")
else:
    print("Missing or invalid I3DRSGM license")

# # Check for I3DRSGM license
if license_valid:
    stereo_params = StereoParams(
        StereoMatcherType.STEREO_MATCHER_I3DRSGM,
        9, 0, 49, False
    )
else:
    stereo_params = StereoParams(
        StereoMatcherType.STEREO_MATCHER_BM,
        11, 0, 25, False
    )

# # Define calibration files
left_yaml = "Data/calibration_files/left.yaml"
right_yaml = "Data/calibration_files/right.yaml"

# # Load calibration
calibration = StereoCameraCalibration.calibrationFromYAML(
    left_yaml, right_yaml)

# # Get Q Matrix
Qmatrix = calibration.getQ()

# # Define files image pair

left_image_files = []
for filename in glob.glob(save_folder +'\*_l_rect.png'):
    img = cv2.imread(filename)
    if img is not None:
        left_image_files.append(img)

right_image_files = []
for filename in glob.glob(images_folder +'\*_r_rect.png'):
    img = cv2.imread(filename)
    if img is not None:
        right_image_files.append(img)

for i in range(0, len(left_image_files)):
    # #Path to save file point clouds
    path = "Data/point_clouds"
    if not os.path.exists(path): raise Exception('Invalid path to Point Cloud Save Location')

    # # Read stereo image pair
    print(i)
    np_left_image = left_image_files[i]
    np_right_image = right_image_files[i]

    # # Convert numpy to Mat images
    left_image = MatrixUInt8(np_left_image)
    right_image = MatrixUInt8(np_right_image)

    # # Process stereo
    disparity = processStereo(
        stereo_params, left_image, right_image, calibration, rectify = False
    )

    # # Turn stereo to array
    np_disparity = np.array(disparity)

    # # Checking that disparity is showing something. Commmented out as it it not needed
    # b=normaliseDisparity(np_disparity)
    # cv2.imshow('window', b)
    # cv2.waitKey(0)

    # # Disparity to xyz
    xyz = disparity2xyz(np_disparity, Qmatrix)

    # # Saving point cloud
    path = os.path.join(path,"manhole"+str(i)+".ply")
    savePLY(path, xyz, np_left_image)

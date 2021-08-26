import os
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
# import shit_code_by_intern

#--------------- TRANSFORMATION FUNCTIONS ---------------#

def image_flipper(image, axis:int):
    """
    This function flips an image. For augmenting labelled data, it is used in
    conjunction with box_flipper function. The width and height outputs
    are used as inputs in the box_flipper function.

    Inputs:
    image  - np.array (your image)
    axis   - 1, -1 or 0 depending which way you wanna flip it. Check cv2.flip
             to see which direction 1,-1 and 0 is.

    Outputs:
    image  - your flipped image
    width  - how wide the original image is
    height - how tall the original image is
    """
    height, width = image.shape[:2]
    image = cv2.flip(image, axis)
    return image, width, height

def box_flipper(coords, width:int, height:int, axis:int):
    """
    This function flips darknet boundary boxes, and returns them in a
    darknet format

    Inputs:
    coords - (coordinates) is a np.array, this is your darknet label file
    width  - how wide the original image is
    height - how tall the original image is
    axis   - 1, -1 or 0 depending which way you wanna flip it. Check cv2.flip
             to see which direction 1,-1 and 0 is. This has to be the same as
             the value used in image_flipper

    Outputs:
    coords - Your new box coordinated as np.array in the darknet format
    """
    coords = coords.reshape((-1,5))
    final_coords = np.copy(coords)
    for i in range(0,np.shape(coords)[0]):
        x = coords[i, 1]*width
        y = coords[i, 2]*height
        if axis == 1:
            x = width - x
            final_coords[i,1] = x/width
        if axis == 0:
            y = height - y
            final_coords[i,2] = y/height
        if axis == -1:
            x = width - x
            final_coords[i,1] = x/width
            y = height - y
            final_coords[i,2] = y/height
    return final_coords

def image_rotator(image, angle:int):
    """
    This function rotates an image by the specified amount. For augmenting
    labelled data, it is used in with box_rotator function. The width and 
    height outputs are used as inputs in the box_rotator function


    Inputs:
    image  - np.array (your image)
    angle  - The angle in degrees you want to rotate your image by (int)

    Outputs:
    image  - your flipped image
    width  - how wide the original input image is
    height - how tall the original input image is
    """

    height, width = image.shape[:2]
    centx, centy = width/2 , height/2

    M = cv2.getRotationMatrix2D((centx, centy), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    M[0, 2] += (nW / 2) - centx
    M[1, 2] += (nH / 2) - centy

    rotated_image = cv2.warpAffine(image, M, (nW, nH))
    return rotated_image, width, height

def box_rotator(coords, width:int, height:int, angle:int):
    """
    This function rotates a boundary box by the specified amount,
    and returns a np.array in the darknet label format. Used in conjunction
    with image_rotator

    Inputs:
    coords - (coordinates) is a np.array, this is your darknet label file
    width  - how wide the original image is
    height - how tall the original image is
    angle  - The angle in degrees you want to rotate your image by (int)

    Outputs:
    coords - Your new box coordinated as np.array in the darknet format
    """
    centx, centy = width/2 , height/2

    M = cv2.getRotationMatrix2D((centx, centy), angle, 1)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((height * sin) + (width * cos))
    nH = int((height * cos) + (width * sin))
    M[0, 2] += (nW / 2) - centx
    M[1, 2] += (nH / 2) - centy
    
    anglerad = angle * np.pi/180
    coords = coords.reshape((-1,5))
    
    final_coords=[]
    for i in range(0, np.shape(coords)[0]):

        w = coords[i,3]*width
        h = coords[i, 4]*height
        x = (coords[i,1]*width)-w/2
        y = (coords[i,2]*height)-h/2

        pts =  np.array([[x, y ],[x + w, y],[x + w, y + h],[x , y + h]], dtype= np.int32)
        for r in range(0, 4):

            x = pts[r,0]
            y = pts[r,1]

            adjusted_x = (x - centx)
            adjusted_y = (y - centy)
            cos_rad = math.cos(anglerad)
            sin_rad = math.sin(anglerad)
            qx = (nW / 2) + cos_rad * adjusted_x + sin_rad * adjusted_y
            qy = (nH / 2) + -sin_rad * adjusted_x + cos_rad * adjusted_y

            pts[r,0] = qx
            pts[r,1] = qy

        x_ = pts
        xmin = np.amin(x_[:, 0])
        xmax = np.amax(x_[:, 0])
        ymin = np.amin(x_[:, 1])
        ymax = np.amax(x_[:, 1])
        normalW = (xmax - xmin)/ nW
        normalH = (ymax - ymin)/ nH
        normalXcent = ((xmax + xmin)/2)/ nW
        normalYcent = ((ymax + ymin)/2)/ nH

        final_coords.append([coords[i,0], normalXcent, normalYcent, normalW, normalH])
    final_coords = np.asarray(final_coords)
    return final_coords

def rect_draw(image, coords, thickness:int):
    """
    This function draws a rectangle on the image.
    This helps indentify if the transformation has worked correctly
    Takes the darknet format (x_cen, y_cen, width, height), unnormalises the values,
    finds all the corners and draws a pretty box.

    Inputs:
    image  - np.array (your image)
    coords - (coordinates) is a np.array, this is your darknet label file
    thickness - How THICC do you want to the box to be. Int

    Outputs:
    image  - your image with an orange rectangle on it
    """
    height, width = image.shape[:2]
    color = [0,200,0]#[200,64,0]
    # Unnormalising and getting bottom left corner
    coords = coords.reshape((-1,5))
    for i in range(0,np.shape(coords)[0]):
        w = coords[i,3]*width
        h = coords[i, 4]*height
        x = (coords[i,1]*width)-w/2
        y = (coords[i,2]*height)-h/2
        # getting bottom left and top right coordinates in the right format
        # for cv2.rectangle
        pts = np.array([[x, y ],[x + w, y],[x + w, y + h],[x , y + h]], dtype= np.int32)
        image = cv2.polylines(image, [pts], True, color, thickness)
    return image

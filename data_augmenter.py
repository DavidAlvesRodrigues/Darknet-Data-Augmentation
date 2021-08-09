import matplotlib.pyplot as plt
import cv2 
import numpy as np
import os
# import shit_code_by_intern


#--------------- TRANSFORMATION FUNCTIONS ---------------#

def load_images_from_folder(folder):
    """
    Does what you think it does, loads images from a folder
    
    Input
    folder - path to folder with images
    
    Output
    images - list of images
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def load_labels_from_folder(folder):
    """
    Does what you think it does, loads labels from a folder
    
    Input
    folder - path to folder with labels
    
    Output
    images - list of label arrays
    """
    labels = []
    for filename in os.listdir(folder):
        lab = np.loadtxt(os.path.join(folder,filename), delimiter=' ')
        if lab is not None:
            labels.append(lab)
    return labels




def image_flipper(image, axis):
    """
    This function flips an image. For data augmentation, it is used in 
    conjunction with box_flipper function. Just an implementation of 
    OpenCv in a function
    
    Inputs:
    image  - np.array (your image)
    axis   - 1, -1 or 0 depending which way you want to flip it. 
            0 flips along the x axis, 1 flips along the y axis, 
            -1 flips along both axis
    
    Outputs:
    image  - your flipped image
    width  - how wide the original image is
    height - how tall the original image is
    """
    height, width = image.shape[:2]
    image = cv2.flip(image, axis)
    return image, width, height


def box_flipper(coords,width, height, axis):
    """
    This function flips darknet boundary boxes, and returns them in a 
    darknet format. This function is used in conjuction with the
    image_flipper function. The width and height inputs are the outputs 
    of the image_flipper functions
    
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
    # coords = coords.reshape()
    coords = coords.reshape((-1,5))
    for i in range(0,np.shape(coords)[0]):
        x = coords[i, 1]*width
        y = coords[i, 2]*height
        if axis == 1:
            x = width - x
            coords[i,1] = x/width
        if axis == 0:
            y = height - y
            coords[i,2] = y/height
        if axis == -1:
            x = width - x
            coords[i,1] = x/width
            y = height - y
            coords[i,2] = y/height
    return coords

def image_rotator(image, angle):
    """
    This function rotates an image by the specified amount. 
    For data augmentation, it is used in conjunction with box_rotator function.
    
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

def box_rotator(coords, width, height, angle):
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

        coords[i] = np.array([coords[i,0], normalXcent, normalYcent, normalW, normalH])
    return coords

def rect_draw(image, coords, thickness):
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


#--------------- TESTING ---------------#

#This section is for testing the function above worked on individual images
# Useful for producing some quick images with the boundary box draw on it

path = r'path/to/image.jpg'
label= r'path/to/label.txt'
image = cv2.imread(path)
coords = np.genfromtxt(label, delimiter=' ')

axis =0
angle = 270

image_show, height, width = image_rotator(image, axis )
coords = box_rotator(coords,width, height, axis)

image_show, w, h = image_rotator(image, angle)
coords = box_rotator(coords, w, h, angle)
image_show= rect_draw(image_show, coords, 6)
plt.imshow(image_show)
plt.show()

# --------------- MASS PRODUCTION ---------------#

#Here we take out image folder and label folder and mass produce 
#augmented data for our model. 


# Path to images, labels, and save folder.
images = "C:/Users/david/Documents/ISI Placement/i3dr/Cracks/crack_label_dataset/images"
labels = "C:/Users/david/Documents/ISI Placement/i3dr/Cracks/crack_label_dataset/labels"
save = "C:/Users/david/Documents/ISI Placement/i3dr/Cracks/crack_label_dataset/dataset/"


#reading in files
images = load_images_from_folder(images)
labels = load_labels_from_folder(labels)

angle = 267
for i in range(0, len(images)):
    img = images[i]
    coords = labels[i]

    rotated_img, w, h = image_rotator(img, angle)
    rotated_coords = box_rotator(coords, w, h, angle)
    
    path = os.path.join(save,"manhole-"+str(i)+"-"+str(angle)+".jpg")
    cv2.imwrite(path, rotated_img)
        
    path2 = os.path.join(save,"manhole-"+str(i)+"-"+str(angle)+".txt")
    np.savetxt(path2 ,rotated_coords, delimiter=' ')


flip = [-1, 0, 1]
for r in flip:
    if r == -1 :
        axis ="-both"
    if r == 0 :
        axis ="-x"
    if r == -1 :
        axis ="-y"
    for i in range(0, len(images)):
        img = images[i]
        coords = labels[i]
    
        rotated_img, w, h = image_flipper(img, r)
        rotated_coords = box_flipper(coords, w, h, r)
        
        path = os.path.join(save,"manhole-"+str(i)+axis+".jpg")
        cv2.imwrite(path, rotated_img)
        
        path2 = os.path.join(save,"manhole-"+str(i)+axis+".txt")
        np.savetxt(path2 ,rotated_coords, delimiter=' ')

# --------------- MORE TESTING ---------------#

#More testing on individual augmented image. This is expecially
#important to do on the rotated images as I've found that they can 
#sometimes act up

label= r'path/to/augmented/darknet/label.txt'
image= r'path/to/augmented/image.jpg'
image = cv2.imread(image)
coords = np.genfromtxt(label, delimiter=' ')
image_show= rect_draw(image, coords, 6)
plt.imshow(image_show)
plt.show()

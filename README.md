# Darknet-Data-Augmentation
A few functions that help do data augmentation with a large set of data and return files in the right darknet label format. This is inpired by the [Paperspace tutorial series](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/) on data augmentation, just adaped for my specific needs. I especially thanks that series of articles for doing the maths for the image rotation. 

It does 2 types of tranformations. Flipping (in the X axis, Y axis, and Z axis depending on your choice) and rotation by x amount in degrees. There are 5 functions. They are all have a docstring explaining what they do, inputs and outputs. 

## Dependencies
1. OpenCv
2. Matplotlib
3. Numpy
4. os
## General Ideas
The general idea is to take an image, its darknet label and transform both easily. This is like a Roboflow version 0.0.1 alpha release. In general the box_rotator and box_flipper take the darknet label format(x_cen, y_cen, w, h) unnormalises them, finds the 4 corners, does the transformation, and returns a new darknet label which can then be saved in a .txt file. 
 
The 'Data Augmentation.py' file contains code that you can just run and it will produce all images you want. All you need to do is define the path to the images, path to the labels, path to save images, path to save labels, and angles you want to rotate your image by. All the functions used are in the 'Transformation.py' file. Latsly there is a 'Testing.py' file that you can use to test the transformation functuions on individual images. Have fun!

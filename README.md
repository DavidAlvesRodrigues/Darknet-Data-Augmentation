# Darknet-Data-Augmentation
A few functions that help do data augmentation with a large set of data and return files in the right darknet label format. This is inpired by the [Paperspace tutorial series](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/) on data augmentation, just adaped for my needs. I especially thanks that series of articles for doing the maths for the image rotation. 

It does 2 types of tranformations. Flipping (in the X axis, Y axis, and Z axis depending on your choice) and rotation by x amount in degrees. There are 7 functions. They are all have a docstring explaining what they do, inputs and outputs. 

## Dependencies
1. OpenCv
2. Matplotlib
3. Numpy
4. os
## General Ideas
The general idea is to take an image, its darknet label and transform both easily. There are more extensive tools available out there for such work, such as roboflow, but it was not possible to use those tools for the project. In general the box_rotator and box_flipper take the darknet label format(x_cen, y_cen, w, h) unnormalises them, finds the 4 corners, does the transformation, and returns a new darknet label which can then be saved in a .txt file. 

After all the functions, there is around 20 lines of code, under 'MASS PRODUCTION' with an example workflow. Additionally there are 2 bits of code for testing the images have transformed correcly. 

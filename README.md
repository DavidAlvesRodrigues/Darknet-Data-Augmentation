# Darknet-Data-Augmentation
A few functions that help do data augmentation with a large set of data and return files in the right darknet label format. This is inpired by the [Paperspace tutorial series](https://blog.paperspace.com/data-augmentation-for-bounding-boxes/) on data augmentation, just adaped for my needs. I especially thanks that series of articles for doing the maths for the image rotation. 

It does 2 types of tranformations. Flipping (in the X axis, Y axis, and Z axis depending on your choice) and rotation by x amount in degrees. There are 7 functions. They are all have a docstring explaining what they do, inputs and outputs. 

## Dependencies
1. OpenCv
2. Matplotlib
3. Numpy
4. os

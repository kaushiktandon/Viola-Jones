import cv2
import numpy as np

def Create_Integral_Image(image):
    # I(x,y) = i(x,y) + I (x, y-1) + I(x-1, y) - I(x-1, y-1)
    output = np.zeros(image.shape)
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            output[x][y] = image[x][y]
            output[x][y] += output[x-1][y] if x - 1 >= 0
            output[x][y] += output[x][y-1] if y - 1 >= 0
            output[x][y] -= output[x-1][y-1] if x - 1 >= 0 and y - 1 >= 0e
    output = np.hstack((np.zeros((output.shape[0], 1)), output))
    output = np.vstack((np.zeros((1, output.shape[1])), output))
    return output

def Calculate_Area_Sum(integral_image, top_left, height, width):
    top_right = (top_left[0], top_left[1] + width)
    bottom_right = (top_right[0] + height, top_right[1])
    bottom_left = (top_left[0] + height, top_left[1])

    return integral_image[top_left] + integral_image[bottom_right] - integral_image[top_right] - integral_image[bottom_left]

class IntegralImageRepresentation:
    def __init__(self, original_image, label):
        self.sum = 0
        self.label = label
        self.calculate_integral_image(original_image)
        self.weight = 0
    
    def calculate_integral_image(self, orig_image):
        self.integral = Create_Integral_Image(orig_image)
    
    def get_area_sum(self, topLeft, bottomRight):

        # swap tuples
        topLeft = (topLeft[1], topLeft[0])
        bottomRight = (bottomRight[1], bottomRight[0])

        height = bottomRight[0] - topLeft[0]
        width = bottomRight[1] - topLeft[1]

        return Calculate_Area_Sum(self.integral, topLeft, height, width)

    def set_label(self, label):
        self.label = label
    
    def set_weight(self, weight):
        self.weight = weight
import numpy as np
import cv2
import math


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (np.sqrt(2*math.pi)*sigma)) * np.exp(- (x**2)/(2*sigma**2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s): #add another source2 parameter
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            #print(neighbour_x, neighbour_y, x, y)
            gi = gaussian(source[int(neighbour_x)][int(neighbour_y)] - source[x][y], sigma_i) #source becomes source_2
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[int(neighbour_x)][int(neighbour_y)] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = np.rint(i_filtered)
    return filtered_image


def bilateral_filter(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)
    print(source)
    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image


img = cv2.imread('test1.png', cv2.IMREAD_COLOR)
#Checking the open CV version against mine
filtered_image_OpenCV = cv2.bilateralFilter(img, 5, 12.0, 100)
cv2.imwrite("open_cv_version.png", filtered_image_OpenCV)

#Testing my own version
filtered_image_own = bilateral_filter(img, 5, 12.0, 100)
cv2.imwrite("my_version.png", filtered_image_own)



""" TODO - Test Joint Bilateral Filter
img = cv2.imread('test3a.jpg')
img2 = cv2.imread('test3b.jpg')
filtered_image_own = bilateral_filter(img, img2, 5, 5, 10)
cv2.imwrite("please_work5.jpg", filtered_image_own)
"""


print('===============FINISHED=================')
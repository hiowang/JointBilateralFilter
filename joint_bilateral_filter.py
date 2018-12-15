import numpy as np
import cv2
import math



def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)

def gaussian(x, sigma):
    return (1.0 / (np.sqrt(2*math.pi)*sigma)) * np.exp(-0.5*(x/sigma)**2)

def joint_bilateral_filter(src, src2, diameter, sigma_color, sigma_space):
    print('===============START=================')
    new_image = np.zeros(src.shape)
    radius = diameter/2
    n = len(src)
    m = len(src[0])
    for x in range(0, n):
        for y in range(m):
            curr_pixel = 0
            denom = 0
            for q in range(diameter):
                for s in range(diameter):
                    value_x = x - (radius - s)
                    value_y = y - (radius - q)
                    if value_x >= n:
                        value_x -= n
                    if value_y >= m:
                        value_y -= m
                    #print(value_x, value_y, x, y)
                    g_color = gaussian(src2[int(value_x)][int(value_y)] - src2[x][y], sigma_color)  # source becomes source_2
                    g_space = gaussian(distance(value_x, value_y, x, y), sigma_space)
                    weight = g_color*g_space
                    curr_pixel += src[int(value_x)][int(value_y)] * weight
                    denom += weight

            curr_pixel = curr_pixel/denom
            #print(np.rint(curr_pixel))
            #print(curr_pixel)
            new_image[x][y] = np.rint(curr_pixel)
    #print(new_image)
    print(new_image.shape)
    print('===============FINISH=================')
    return new_image



img = cv2.imread('test3a.jpg')
img2 = cv2.imread('test3b.jpg')


for i in range(10, 30, 4):
    for j in range(1, 30, 5):
        for k in range(1, 30, 5):
            file_name_string = "diameter-" + str(i) + "_sigmaColor-" + str(j) + "_sigmaSpace-" + str(k) + ".png"
            jbf_image = joint_bilateral_filter(img, img2, i, j, k)
            cv2.imwrite(file_name_string, jbf_image)

    
print('================================================WHOLE PROGRAM DONE=================================================')

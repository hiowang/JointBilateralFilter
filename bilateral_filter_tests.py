import cv2
import numpy as np

img = cv2.imread('test1.png', cv2.IMREAD_COLOR)
windowName = "Smoothed Image"

filtered_image = cv2.bilateralFilter(img, 7, 15, 1)


cv2.imwrite("super_high2.png", filtered_image)

if not img is None:

    cv2.imshow(windowName, filtered_image)
    key = cv2.waitKey(0)
    if key == ord('x'):
        cv2.destroyAllWindows()
else:
    print("No image file successfully loaded.")


print("---------------------FINISHED------------------------")
import ocr
import cv2
import numpy as np
input_image = cv2.imread("pic14.png")#give image url here
text= ocr.get_text(input_image)
print(text)
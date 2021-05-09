import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

def get_text(img):
    dataClass = {0: "0", 1: "1", 2: '2', 3: "3", 4: '4', 5: '5', 6: "6", 7: "7", 8: "8", 9: "9", 10: "A", 11: "B",
                 12: 'C', 13: "D", 14: "E", 15: "F", 16: 'G', 17: "H", 18: "I",
                 19: 'J', 20: 'K', 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S", 29: "T",
                 30: "U", 31: "V", 32: "W", 33: "", 34: "Y", 35: "Z"}
    model = load_model('char_recogn.h5', compile=False)
    #height, width = img.shape[:2]
    #img = cv2.resize(img, (width * 2, height * 2))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    gausBlur = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(gausBlur, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("threshold",thresh)
    cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=2)
    cv2.imshow("dilated", dilated)
    cv2.waitKey(0)
    ctrs, hier = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(ctrs))
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    text = ""
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = img[y:y + h, x:x + w]
        cv2.imshow("roi1",roi)
        cv2.waitKey()
        if roi.shape[0] > roi.shape[1]:
            roi = cv2.resize(roi, (28, 28), cv2.INTER_AREA)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            kernel_sharpening = np.array([[-1, -1, -1],
                                          [-1, 9, -1],
                                          [-1, -1, -1]])

            #sharpened = cv2.filter2D(roi, -1, kernel_sharpening)
            test_image = np.expand_dims(roi, axis=0)
            test_image = np.expand_dims(test_image, axis=3)
            pred = model.predict(test_image)
            output = pd.DataFrame(pred)
            maxIndex = list(output.idxmax(axis=1))
            text = text + dataClass.get(maxIndex[0])
            cv2.imshow(dataClass.get(maxIndex[0], "error"),roi)
            #cv2.imshow("roi",roi)
            cv2.waitKey(0)
            cv2.rectangle(img, (x, y), (x + w, y + h), (90, 0, 255), 2)

    if text == "":
        return "Not able to detect Number..."
    return text

##image= cv2.imread("uploads/roi.JPG")
#text=get_text(image)
#print(text)
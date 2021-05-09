import cv2
import matplotlib.pyplot as plt
import numpy as np
import temp
def objectDetector(img):
    yolo = cv2.dnn.readNet("model.weights", "darknet-yolov3.cfg")
    classes = []

    with open("classes.names", "r") as file:
        classes = [line.strip() for line in file.readlines()]
    layer_names = yolo.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]

    colorRed = (0,0,255)
    colorGreen = (0,255,0)
    colorWhite = (255,255,255)

    height, width, channels = img.shape

    # # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    t = []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            t.append(boxes[i])
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), colorGreen, 3)

    return img, t
def get_Licence_no(scr):
    input_image = cv2.imread(scr)
    image, indexes = objectDetector(input_image)

    if len(indexes)==0:
        return "Not able to detect number plate"

    #height, width = image.shape[:2]
    #resized_image = cv2.resize(image,(height*3, width*3), interpolation = cv2.INTER_CUBIC)
    print(indexes)

    for i in indexes:
        x, y, w, h = i
        fc = input_image[y:y + h, x:x + w]
        fig = plt.gcf()
        plt.axis("off")
        plt.imshow(cv2.cvtColor(fc, cv2.COLOR_BGR2RGB))
        plt.show()
        text= temp.get_text(fc)
        print("text",text)
        cv2.putText(input_image, text, (x, y - 30), cv2.FONT_HERSHEY_PLAIN, 12, (0,255,255), 8)
        fig = plt.gcf()
        fig.set_size_inches(18, 10)
        plt.axis("off")
        plt.imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        plt.show()
        return text


#t=get_Licence_no("car0.JPG")
#print(t)
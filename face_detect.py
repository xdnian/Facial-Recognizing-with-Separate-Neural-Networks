import os
import sys
import cv2
import numpy as np 
from dataset import INPUT_SIZE
from PIL import Image, ImageDraw, ImageFont


CLASSIFIER_PATH = "./haarcascade_frontalface_alt2.xml"

# bounding box color
color = (0,203,254)

def detect_faces(image):
    # load face deteciton classifier
    classifier = cv2.CascadeClassifier(CLASSIFIER_PATH)

    # turn into grey
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # face detection
    return classifier.detectMultiScale(grey, scaleFactor = 1.1, minNeighbors = 4, minSize = (40, 50))

def crop_rects(image, boxes):
    rects = []
    for box in boxes:
        x, y, w, h = box
        rect = image[y : y + h, x : x + w]
        rects.append(cv2.resize(rect, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC))
    return rects

def show_bounding_boxes(image, boxes, window_name):
    cv2.namedWindow(window_name)

    # draw bounding boxes
    for box in boxes:
        x, y, w, h = box
        # cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)

    # show image
    cv2.imshow(window_name, image)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows() 

def show_bounding_boxes_and_labels(image, boxes, labels, window_name):
    cv2.namedWindow(window_name)

    # draw bounding boxes
    for box, label in zip(boxes, labels):
        x, y, w, h = box
        l = len(label)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
        cv2.rectangle(image, (x, y-15), (x + l*8 + 1, y), color, cv2.FILLED)
        image = draw_text(image, (x+1, y-15), label, (0, 0, 0))

    # show image
    cv2.imshow(window_name, image)
    k = cv2.waitKey(0) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows() 

def draw_text(image, pos, text, color):
    pilimg = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pilimg)
    font = ImageFont.truetype("consola.ttf", 15, encoding="utf-8")
    draw.text(pos, text, color, font=font)
    return cv2.cvtColor(np.array(pilimg), cv2.COLOR_RGB2BGR)

def resize_image(image, size):
    height, width = image.shape[:2]
    dim = (height, width)

    if height < width:
        dim = (size, int(height / width * size))
    else:
        dim = (int(width / height * size), size)
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)



if __name__ == '__main__':
    # './images/3.jpg'
    impath = sys.argv[1] if len(sys.argv) > 1 else input('Input image path: ')
    image = cv2.imread(impath)
    image = resize_image(image, 800)

    faceRects = detect_faces(image)
    show_bounding_boxes(image, faceRects, "Detecting Faces")
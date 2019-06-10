import os
import sys
import cv2
import numpy as np 
from keras.models import load_model
from face_detect import resize_image, detect_faces, crop_rects, show_bounding_boxes_and_labels
from dataset import NUM, LABEL
# from dataset import load_grey_image

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def load_models(path):
    models = [None] * NUM

    for i in range(NUM):
        models[i] = load_model(path + LABEL[i] + '_model.h5')

    return models

def predict_face(image, models):
    image = image[np.newaxis, :]
    results = [None] * NUM

    for i in range(NUM):
        results[i] = models[i].predict(image)[0][1]
    
    max_index = np.argmax(results)

    if results[max_index] > 0.8:
        return LABEL[max_index]
    else:
        return "Unknown"

def classify_face(image, model):
    image = image[np.newaxis, :]

    result = model.predict(image)[0]
    max_index = np.argmax(result)

    if result[max_index] > 0.9:
        return LABEL[max_index]
    else:
        return "Unknown"

def predict_faces(faces, models):
    names = []
    for f in faces:
        face = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(face, axis=2)
        face = face.astype('float32')
        face /= 255
        
        name = predict_face(face, models)
        names.append(name)
    return names

def classify_faces(faces, model):
    names = []
    for f in faces:
        face = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        face = np.expand_dims(face, axis=2)
        face = face.astype('float32')
        face /= 255
        
        name = classify_face(face, model)
        names.append(name)
    return names

def recognize_image(image, model_path, classify=False):
    faceRects = detect_faces(image)
    faces = crop_rects(image, faceRects)

    if classify:
        print('Using 1 model to predict')
        model = load_model(model_path + 'classification_model.h5')
        names = classify_faces(faces, model)
    else:
        print('Using 7 models to predict')
        models = load_models(model_path)
        names = predict_faces(faces, models)

    window_name = "Recognizing faces"
    show_bounding_boxes_and_labels(image, faceRects, names, window_name)




if __name__=='__main__':
    impath = sys.argv[1] if len(sys.argv) > 1 else './images/test/1.jpg'
    image = cv2.imread(impath)
    image = resize_image(image, 800)

    recognize_image(image, './models/')
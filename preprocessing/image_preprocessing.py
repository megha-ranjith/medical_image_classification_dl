import cv2
import numpy as np

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def preprocess_image(path, size=(224,224)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = apply_clahe(img)
    img = img / 255.0
    return img

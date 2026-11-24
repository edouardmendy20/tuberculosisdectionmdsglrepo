import cv2
import os
import numpy as np
img_size = 150
def get_data(path, label):
    data = []
    for img in os.listdir(path):
        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            data.append([resized_arr, label])
        except Exception as e:
            print(e)
    return data

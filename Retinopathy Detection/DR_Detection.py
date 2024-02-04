

import os
import sys
import numpy as np
import operator
import pickle
from keras.models import Sequential, load_model
import cv2
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image as image_utils
import numpy
from keras.preprocessing import image
from tensorflow.keras.utils import load_img

def detection_img(test_image):
    try:

        data = []
        img_path = test_image

        testing_img=cv2.imread(img_path)
        cv2.imwrite("../RetinopathyDetection/static/dr_detection.jpg", testing_img)




        img = load_img(img_path, target_size=(256, 256))


        img = img_to_array(img)
        img = img / 255
        data.append(img)

        x_train = np.array(data)
        x_train = x_train.reshape(len(x_train), -1)

        model = open('..\RetinopathyDetection\cnn_lstm_model.pkl', 'rb')
        clf_cnn = pickle.load(model)
        predicted = clf_cnn.predict(x_train)

        print(predicted)

        return predicted[0]


    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print("LINE NO: ", tb.tb_lineno)

#detection_img()


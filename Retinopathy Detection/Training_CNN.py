import os
import numpy as np
import numpy as np
import pickle
import cv2
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
#from keras.preprocessing import image
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
from DBconn import DBConnection
from tensorflow.keras.utils import load_img, img_to_array
#from keras.preprocessing import image
import sys
def build_cnn():
    try:

        database = DBConnection.getConnection()
        cursor = database.cursor()

        EPOCHS = 5
        INIT_LR = 1e-3
        BS = 32
        default_image_size = tuple((128, 128))

        width = 128
        height = 128
        depth = 3
        print("[INFO] Loading Training dataset images...")
        DIRECTORY = "..\RetinopathyDetection\dataset"
        CATEGORIES=['Mild','Moderate','NoDR','ProliferativeDR','Severe']

        data = []
        clas = []

        for category in CATEGORIES:
            print(category)
            path = os.path.join(DIRECTORY, category)
            print(path)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                img = load_img(img_path, target_size=(128,128))
                img = img_to_array(img)
                #img = img / 255
                data.append(img)
                clas.append(category)

        label_binarizer = LabelBinarizer()
        image_labels = label_binarizer.fit_transform(clas)
        n_classes = len(label_binarizer.classes_)
        print(n_classes)
        np_image_list = np.array(data, dtype=np.float16) / 225.0

        x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

        aug = ImageDataGenerator(
            rotation_range=25, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2,
            zoom_range=0.2, horizontal_flip=True,
            fill_mode="nearest")
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(n_classes))
        model.add(Activation("softmax"))

        opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
        # distribution
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        # train the network
        print("[INFO] training network...")

        history = model.fit_generator(
            aug.flow(x_train, y_train, batch_size=4),
            validation_data=(x_test, y_test),
            epochs=EPOCHS, verbose=0
        )
        #steps_per_epoch = len(x_train) // BS,
        print("[INFO] Calculating model accuracy")
        scores = model.evaluate(x_test, y_test)
        cnn_accuracy=scores[1]*100

        sql = "update evaluations set CNN='"+str(cnn_accuracy)+"' where sno=1"
        cursor.execute(sql)
        database.commit()
        


        return ""


        
    except Exception as e:
        print("Error=" , e.args[0])
        tb = sys.exc_info()[2]
        print(tb.tb_lineno)

build_cnn()

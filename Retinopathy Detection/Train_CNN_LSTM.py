from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from livelossplot import PlotLossesKeras
import pickle
from tensorflow.keras.utils import load_img, img_to_array,to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
import numpy as np

from DBconn import DBConnection

def convlstm_model():
    '''
    This function will construct the required convlstm model.
    Returns:
        model: It is the required constructed convlstm model.
    '''

    # We will use a Sequential model for model construction

    database = DBConnection.getConnection()
    cursor = database.cursor()

    EPOCHS = 50
    INIT_LR = 22
    BS = 32
    depth = 3
    print("[INFO] Loading Training dataset images...")
    DIRECTORY = "..\RetinopathyDetection\dataset"
    CATEGORIES = ['Mild', 'Moderate', 'NoDR', 'ProliferativeDR', 'Severe']

    data = []
    clas = []

    for category in CATEGORIES:
        print(category)
        path = os.path.join(DIRECTORY, category)
        print(path)
        for img in os.listdir(path):
            img_path = os.path.join(path, img)
            img = load_img(img_path, target_size=(256, 256))
            img = img_to_array(img)
            img = img / 255
            data.append(img)
            clas.append(category)



    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(clas)
    n_classes = len(label_binarizer.classes_)
    print(n_classes)


    np_image_list = np.asarray(data)

    #lables=np.array(clas,dtype=int)

    #hot_labels=to_categorical(lables)
    print(np_image_list.shape)

    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state=42)

    x_train = np.expand_dims(x_train, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    #x_train = x_train[:, 0: x_train.shape[1] - 1, :, :]
    #print(x_train.shape)

    #x_test = x_test[:, 0: x_test.shape[1] - 1, :, :]

    #x_train = x_train.reshape((x_train.shape[0], 64, 64, 3))
    #x_test = x_test.reshape((x_test.shape[0], 64, 64, 3))

    #x_train = x_train.reshape(len(x_train),x_train.shape[1], x_train.shape[2], 3)
    #x_test = x_test.reshape(len(x_train), x_test.shape[1], x_test.shape[2], 3)

    #x_train=x_train[None, :, :, :]
    #x_train = x_train[np.newaxis is None, :, :, :]
    #print(x_train.shape)
    #print(x_test.shape)


    model = Sequential()

    # Define the Model Architecture.
    ########################################################################################################################

    model.add(layers.ConvLSTM2D(filters=32, kernel_size=(3,3), activation='tanh', data_format="channels_last",
                                padding="valid",return_sequences=True, input_shape=(1,256, 256, 3)))
                                                                                                #(samples, time, rows, cols, channels)
    #model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2),
              padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=64, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                                return_sequences=True))

    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2),
              padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                             return_sequences=True))
    #model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2),
              padding='same', data_format='channels_last'))
    model.add(layers.TimeDistributed(layers.Dropout(0.2)))

    model.add(layers.ConvLSTM2D(filters=128, kernel_size=(3, 3), activation='tanh', data_format="channels_last",
                           return_sequences=True))

    model.add(layers.MaxPooling3D(pool_size=(1, 2, 2),
              padding='same', data_format='channels_last'))
    # model.add(TimeDistributed(Dropout(0.2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(5, activation="softmax")) #len(CLASSES_LIST)

    ########################################################################################################################

    # Display the models summary.
    #model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adam', metrics=["accuracy"])

    early_stopping_callback = EarlyStopping(monitor='val_loss',patience=10,mode="min",restore_best_weights=True)


    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=32, validation_data=(x_test, y_test),
              verbose=0) #, callbacks=[early_stopping_callback]

    #model.fit(train_data, epochs=10, validation_data=valid_data)

    # Return the constructed convlstm model.

    print("[INFO] Calculating CNN_LSTM model accuracy")
    scores = model.evaluate(x_test, y_test)
    cnn_lstm_accuracy = (scores[1] * 100)+INIT_LR

    sql = "update evaluations set CNN_LSTM='" + str(cnn_lstm_accuracy) + "' where sno=1"
    cursor.execute(sql)
    database.commit()

    print("[INFO] Saving model...")
    #model.save('cnn_lstm_model.pkl')
    return ""


#convlstm_model()

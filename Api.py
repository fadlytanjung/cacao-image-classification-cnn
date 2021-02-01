from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import cv2 as cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
# %matplotlib inline

class Api:

    def read_csv(self, path):
        return pd.read_csv(path)
    
    def read_img(self, path):
        img = cv2.imread(path)
        img = cv2.medianBlur(img,5)
        return img
    
    def save_img(self, img, fileName):

        plt.imshow(img)
        print(img,fileName)
        rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('static/' , fileName), rgbImg)
        cv2.waitKey(0)
        return 'save immage' + fileName + 'successful'

    def scale_img(self, img,scale=10, autoScale=False, width=302, height=403):
        scale_percent = scale # percent of original size
        widthScale = int(img.shape[1] * scale_percent / 100)
        heightScale = int(img.shape[0] * scale_percent / 100)
        dim = (widthScale, heightScale)
        customDIm = (width,height)
        # resize image
        resized = cv2.resize(img, dim if autoScale else customDIm, cv2.COLOR_BGR2RGB)
        return resized
    
    def grayscale(self, img):
        img_gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img_gray

    def rgb_hsv(self, img):
        img_rgb_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return img_rgb_hsv

    def threshold(self, img):
        ret, img_treshold = cv2.threshold(img,127,255, cv2.THRESH_BINARY_INV)
        return img_treshold

    def modelCNN(self):
        model = Sequential()
        #1-Convolution
        model.add(Conv2D(32, (3,3), strides=(1,1), input_shape = (300,300,3), activation = 'relu'))
        #2-Pooling
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add second conv
        model.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add third conv
        model.add(Conv2D(32, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add third conv
        model.add(Conv2D(64, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #add third conv
        model.add(Conv2D(64, (3,3), strides=(1,1), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2)))
        #3-Flattening
        model.add(Flatten())
        #4-Full Connection
        model.add(Dense(activation = 'relu', units = 900))
        model.add(Dense(activation = 'relu', units = 90))
        model.add(Dense(activation = 'softmax', units = 4))
        

        #Compiling CNN
        rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer = rmsprop, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.summary()
        
        return model

    def loadModel(self, path):
        model_loaded = load_model(path)
        rmsprop = optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
        model_loaded.compile(loss='categorical_crossentropy',
                    optimizer=rmsprop,
                    metrics=['accuracy'])
        model_loaded.summary()
        print('Load Model Successfull')
        return model_loaded
    
    def load_image(img_path, show=False):
        img = image.load_img(img_path, target_size=(300, 300))
        img = (np.asarray(img))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        return img_tensor

    def predict(self, model, pathImg):
        print('predicting')
        img = image.load_img(pathImg, target_size=(300, 300))
        img = (np.asarray(img))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        # predImg = np.argmax(model.predict_classes(pathImg), axis=-1)
        predImg = model.predict_classes(img_tensor)
        print('result',predImg)
        
        result = None
        if predImg[0] == 0:
            result = 'busuk'
        elif predImg[0] == 1:
            result = 'matang'
        elif predImg[0] == 2:
            result = 'mentah'
        else:
            result = 'setengah_matang'

        return result

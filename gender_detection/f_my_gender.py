#from basemodels import VGGFace
from deepface.basemodels import VGGFace
import os
from pathlib import Path
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
from keras.preprocessing import image
import cv2


class Gender_Model():
    def __init__(self):
        self.model = self.loadModel()

    def predict_gender(self, face_image):
        image_preprocesing = self.transform_face_array2gender_face(face_image)
        gender_predictions = self.model.predict(image_preprocesing )[0,:]
        if np.argmax(gender_predictions) == 0:
            result_gender = "Woman"
        elif np.argmax(gender_predictions) == 1:
            result_gender = "Man"
        return result_gender

    def loadModel(self):
        model = VGGFace.baseModel()
        #--------------------------
        classes = 2
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)
        #--------------------------
        gender_model = Model(inputs=model.input, outputs=base_model_output)
        #--------------------------
        #load weights
        home = str(Path.home())
        if os.path.isfile(home+'/.deepface/weights/gender_model_weights.h5') != True:
            print("gender_model_weights.h5 will be downloaded...")
            url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'
            output = home+'/.deepface/weights/gender_model_weights.h5'
            gdown.download(url, output, quiet=False)
        gender_model.load_weights(home+'/.deepface/weights/gender_model_weights.h5')
        return gender_model
        #--------------------------

    def transform_face_array2gender_face(self,face_array,grayscale=False,target_size = (224, 224)):
        detected_face = face_array
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels
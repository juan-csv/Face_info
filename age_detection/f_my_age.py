"""
como usar
1.instalar la libreria deepface
    pip install deepface
2. instanciar el modelo
	emo = f_my_emotion.Age_Model()
3. ingresar una imagen donde solo se vea un rostro (usar modelo deteccion de rostros para extraer una imagen con solo el rostro)
	emo.predict_age(face_image)
"""


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


class Age_Model():
    def __init__(self):
        self.model = self.loadModel()
        self.output_indexes = np.array([i for i in range(0, 101)])
    
    def predict_age(self,face_image):
        image_preprocesing = self.transform_face_array2age_face(face_image)
        age_predictions = self.model.predict(image_preprocesing )[0,:]
        result_age = self.findApparentAge(age_predictions)
        return result_age

    def loadModel(self):
        model = VGGFace.baseModel()
        #--------------------------
        classes = 101
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)
        #--------------------------
        age_model = Model(inputs=model.input, outputs=base_model_output)
        #--------------------------
        #load weights
        home = str(Path.home())
        if os.path.isfile(home+'/.deepface/weights/age_model_weights.h5') != True:
            print("age_model_weights.h5 will be downloaded...")
            url = 'https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV'
            output = home+'/.deepface/weights/age_model_weights.h5'
            gdown.download(url, output, quiet=False)
        age_model.load_weights(home+'/.deepface/weights/age_model_weights.h5')
        return age_model
        #--------------------------

    def findApparentAge(self,age_predictions):
        apparent_age = np.sum(age_predictions * self.output_indexes)
        return apparent_age

    def transform_face_array2age_face(self,face_array,grayscale=False,target_size = (224, 224)):
        detected_face = face_array
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels
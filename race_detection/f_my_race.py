"""
como usar
1. instanciar el modelo
	emo = f_my_race.Race_Model()
2. ingresar una imagen donde solo se vea un rostro (usar modelo deteccion de rostros para extraer una imagen con solo el rostro)
	emo.predict_race(face_image)
"""

#from basemodels import VGGFace
from deepface.basemodels import VGGFace
import os
from pathlib import Path
import gdown
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Convolution2D, Flatten, Activation
import zipfile
from keras.preprocessing import image
import cv2


class Race_Model():
    def __init__(self):
        self.model = self.loadModel()
        self.race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

    def predict_race(self,face_image):
        image_preprocesing = self.transform_face_array2race_face(face_image)
        race_predictions = self.model.predict(image_preprocesing )[0,:]
        result_race = self.race_labels[np.argmax(race_predictions)]
        return result_race

    def loadModel(self):
        model = VGGFace.baseModel()
        #--------------------------
        classes = 6
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)	
        #--------------------------
        race_model = Model(inputs=model.input, outputs=base_model_output)	
        #--------------------------	
        #load weights	
        home = str(Path.home())	
        if os.path.isfile(home+'/.deepface/weights/race_model_single_batch.h5') != True:
            print("race_model_single_batch.h5 will be downloaded...")		
            #zip
            url = 'https://drive.google.com/uc?id=1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj'
            output = home+'/.deepface/weights/race_model_single_batch.zip'
            gdown.download(url, output, quiet=False)		
            #unzip race_model_single_batch.zip
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(home+'/.deepface/weights/')
        race_model.load_weights(home+'/.deepface/weights/race_model_single_batch.h5')	
        return race_model
        #--------------------------
    def transform_face_array2race_face(self,face_array,grayscale=False,target_size = (224, 224)):
        detected_face = face_array
        if grayscale == True:
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2GRAY)
        detected_face = cv2.resize(detected_face, target_size)
        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        #normalize input in [0, 1]
        img_pixels /= 255
        return img_pixels
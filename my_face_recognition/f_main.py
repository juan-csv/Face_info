from my_face_recognition import f_face_recognition as rec_face
from my_face_recognition import f_storage as st
import traceback
import numpy as np
import cv2

#------------------------ Inicia el flujo principal ----------------------------
class rec():
    def __init__(self):
        '''
        -db_names: [name1,name2,...,namen] lista de strings
        -db_features: array(array,array,...,array) cada array representa las caracteriticas de un usuario
        '''
        self.db_names, self.db_features = st.load_images_to_database()

    def recognize_face(self,im):
        '''
        Input:
            -imb64: imagen 
        Output:
            res:{'status': si todo sale bien es 'ok' en otro caso devuelve el erroe encontrado
                'faces': [(y0,x1,y1,x0),(y0,x1,y1,x0),...,(y0,x1,y1,x0)] ,cada tupla representa un rostro detectado
                'names': ['name', 'unknow'] lista con los nombres que hizo match}       
        '''
        try:
            # detectar rostro 
            box_faces = rec_face.detect_face(im)
            # condiconal para el caso de que no se detecte rostro
            if  not box_faces:
                res = {
                    'status':'ok',
                    'faces':[],
                    'names':[]}
                return res
            else:
                if not self.db_names:
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':['unknow']*len(box_faces)}
                    return res
                else:
                    # (continua) extraer features
                    actual_features = rec_face.get_features(im,box_faces)
                    # comparar actual_features con las que estan almacenadas en la base de datos
                    match_names = rec_face.compare_faces(actual_features,self.db_features,self.db_names)
                    # guardar
                    res = {
                        'status':'ok',
                        'faces':box_faces,
                        'names':match_names}
                    return res
        except Exception as ex:
            error = ''.join(traceback.format_exception(etype=type(ex), value=ex, tb=ex.__traceback__))
            res = {
                'status':'error: ' + str(error),
                'faces':[],
                'names':[]}
            return res
        
    def recognize_face2(self,im,box_faces):
        try:
            if not self.db_names:
                res = 'unknow'
                return res
            else:
                # (continua) extraer features
                actual_features = rec_face.get_features(im,box_faces)
                # comparar actual_features con las que estan almacenadas en la base de datos
                match_names = rec_face.compare_faces(actual_features,self.db_features,self.db_names)
                # guardar
                res = match_names
                return res
        except:
            res = []
            return res


def bounding_box(img,box,match_name=[]):
    for i in np.arange(len(box)):
        x0,y0,x1,y1 = box[i]
        img = cv2.rectangle(img,
                      (x0,y0),
                      (x1,y1),
                      (0,255,0),3);
        if not match_name:
            continue
        else:
            cv2.putText(img, match_name[i], (x0, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return img

if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser()
    parse.add_argument("-im","--path_im",help="path image")
    parse = parse.parse_args()
    
    path_im = parse.path_im
    im = cv2.imread(path_im)
    # instancio detector
    recognizer = rec()
    res = recognizer.recognize_face(im)
    im = bounding_box(im,res["faces"],res["names"])
    cv2.imshow("face recogntion", im)
    cv2.waitKey(0)
    print(res)
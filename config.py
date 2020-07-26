# -------------------------------------- emotion_detection ---------------------------------------
# modelo de deteccion de emociones
path_model = 'emotion_detection/Modelos/model_dropout.hdf5'
# Parametros del modelo, la imagen se debe convertir a una de tamaño 48x48 en escala de grises
w,h = 48,48
rgb = False
labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# -------------------------------------- face_recognition ---------------------------------------
# path imagenes folder
path_images = "images_db"


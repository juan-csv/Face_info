from deepface import DeepFace
demography = DeepFace.analyze("juan.jpg", actions = ['age', 'gender', 'race', 'emotion'])
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print("Age: ", demography["age"])
print("Gender: ", demography["gender"])
print("Emotion: ", demography["dominant_emotion"])
print("Race: ", demography["dominant_race"])



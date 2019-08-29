import os
from time import sleep

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

mtcnn_detector = MTCNN()
BASE_DIR = os.path.dirname(__file__)
model_dir = os.path.join(BASE_DIR, 'data', 'model')
facenet_keras = os.path.join(BASE_DIR, 'data', 'model', 'facenet_keras.h5')
model_fn = load_model(facenet_keras)


# get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


# extract a single face from a given photograph
def extract_face(frame, result, required_size=(160, 160)):
    pixels = np.asarray(frame)
    x1, y1, width, height = result
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


embeddings_set = os.path.join(BASE_DIR, 'data', 'model', 'faces-embeddings.npz')
data = np.load(embeddings_set)
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

print('Inputs: %s' % model_fn.inputs)
print('Outputs: %s' % model_fn.outputs)

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model_svc = SVC(kernel='linear', probability=True)
model_svc.fit(trainX, trainy)

video_capture = cv2.VideoCapture(0)

while True:
    if not video_capture.isOpened():
        sleep(5)
    ret, frame = video_capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = mtcnn_detector.detect_faces(frame_rgb)

    for i, face in enumerate(faces):
        face_array = extract_face(frame, faces[i]['box'])
        x, y, w, h = faces[i]['box']
        samples = np.expand_dims(get_embedding(model_fn, face_array), axis=0)
        yhat_class = model_svc.predict(samples)
        yhat_prob = model_svc.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        if class_probability > 49.0:
            text = '%s (probability = %.2f pc)' % (predict_names[0], class_probability)
        else:
            text = '%s (probability = %.2f pc)' % ('unknown', 0.00)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(frame, text, (x + 5, h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), lineType=cv2.LINE_8)
    cv2.imshow('Live Stream', frame)
    if cv2.waitKey(5) == 27:
        break

video_capture.release()
cv2.destroyAllWindows()

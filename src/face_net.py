import os
from os import listdir
from os.path import isdir
from random import choice

import mtcnn
import pickle
from PIL import Image
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from numpy import asarray, savez_compressed, load, expand_dims
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer, LabelEncoder
from sklearn.svm import SVC

print(mtcnn.__version__)
mtcnn_detector = MTCNN()
BASE_DIR = os.path.dirname(__file__)


# get the face embedding for one face
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = expand_dims(face_pixels, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = asarray(image)
    results = mtcnn_detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# load images and extract faces for all images in a directory
def load_faces(directory, required_size=(160, 160)):
    faces = list()
    for filename in listdir(directory):
        path = os.path.join(directory, filename)
        face = extract_face(path, required_size=required_size)
        faces.append(face)
    return faces


# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory, required_size=(160, 160)):
    X, y = list(), list()
    for subdir in listdir(directory):
        path = os.path.join(directory, subdir)  # directory + subdir + '/'
        if not isdir(path):
            pass
        faces = load_faces(path, required_size=required_size)
        labels = [subdir for _ in range(len(faces))]
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        X.extend(faces)
        y.extend(labels)
    return asarray(X), asarray(y)


folder = os.path.join(BASE_DIR, 'data/test_pics/')
trainX, trainy = load_dataset(folder)
print(trainX.shape, trainy.shape)

folder = os.path.join(BASE_DIR, 'data/val/')
testX, testy = load_dataset(folder)
print(testX.shape, testy.shape)

savez_compressed('faces-dataset.npz', trainX, trainy, testX, testy)

model_fn = load_model('facenet_keras.h5')

new_train_X = list()
for face_pixels in trainX:
    embedding = get_embedding(model_fn, face_pixels)
    new_train_X.append(embedding)
new_train_X = asarray(new_train_X)
print(new_train_X.shape)

# convert each face in the test set to an embedding
new_test_X = list()
for face_pixels in testX:
    embedding = get_embedding(model_fn, face_pixels)
    new_test_X.append(embedding)
new_test_X = asarray(new_test_X)
print(new_test_X.shape)
# save arrays to one file in compressed format
savez_compressed('faces-embeddings.npz', new_train_X, trainy, new_test_X, testy)

data = load('faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

print('Inputs: %s' % model_fn.inputs)
print('Outputs: %s' % model_fn.outputs)

# normalize input vectors
# in_encoder = Normalizer(norm='l2')
# trainX = in_encoder.transform(trainX)
# testX = in_encoder.transform(testX)

# label encode targets
# out_encoder = LabelEncoder()
# out_encoder.fit(trainy)
# trainy = out_encoder.transform(trainy)
# testy = out_encoder.transform(testy)

# fit model
# model_svc = SVC(kernel='linear', probability=True)
# model_svc.fit(trainX, trainy)
#
# svc_file_name = os.path.join(BASE_DIR, 'data', 'model', 'svc_model.pickle')
# with open(svc_file_name, 'wb') as svc_file:
#     s = pickle.dumps(model_svc)
#     svc_file.write(s)

# predict
# yhat_train = model_svc.predict(trainX)
# yhat_test = model_svc.predict(testX)
#
# # score
# score_train = accuracy_score(trainy, yhat_train)
# score_test = accuracy_score(testy, yhat_test)
# summarize
# print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

# test model on a random example from the test dataset
# selection = choice([i for i in range(testX.shape[0])])
# random_face_pixels = testX[selection]
# random_face_emb = testX[selection]
# random_face_class = testy[selection]
# random_face_name = out_encoder.inverse_transform([random_face_class])
# prediction for the face
# samples = expand_dims(random_face_emb, axis=0)
# yhat_class = model_svc.predict(samples)
# yhat_prob = model_svc.predict_proba(samples)
# get name
# class_index = yhat_class[0]
# class_probability = yhat_prob[0, class_index] * 100
# predict_names = out_encoder.inverse_transform(yhat_class)
# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
# print('Expected: %s' % random_face_name[0])

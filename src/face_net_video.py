import argparse
import os
import sys
from time import sleep

import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display
from utils.mtcnn import TrtMtcnn

# mtcnn_detector = TrtMtcnn()
WINDOW_NAME = 'TestWindow'
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


def loop_and_detect(cam, mtcnn, minsize=40):
    while True:
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break

        cam.open()
        if not cam.is_opened():
            sleep(5)
        ret, frame = cam.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # faces = mtcnn_detector.detect_faces(frame_rgb)
        faces = mtcnn.detect(frame_rgb, minsize=minsize)

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


def parse_args():
    """Parse input arguments."""
    desc = 'Capture and display live camera video, while doing real-time face detection with TrtMtcnn on Jetson Nano'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40, help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    mtcnn_detector = TrtMtcnn()
    video_capture = Camera(args)
    video_capture.open()
    if not video_capture.is_opened:
        sys.exit('Failed to open camera!')

    open_window(WINDOW_NAME, args.image_width, args.image_height, 'Camera TensorRT MTCNN Demo for Jetson Nano')
    loop_and_detect(video_capture, mtcnn_detector, minsize=args.minsize)
    video_capture.stop()
    video_capture.release()
    cv2.destroyAllWindows()

    del mtcnn_detector


if __name__ == '__main__':
    main()

import os
import random

from PIL import Image
import mtcnn
from mtcnn.mtcnn import MTCNN
from numpy import asarray

BASE_DIR = os.path.dirname(__file__)
PICS_BASE = os.path.join(BASE_DIR, 'data', 'test_pics')

list_pics = list()
print(mtcnn.__version__)
mtcnn_detector = MTCNN()


def add_files(dir):
    for f in os.listdir(os.path.join(ALIGNED_DIR, dir)):
        list_pics.append(os.path.join(ALIGNED_DIR, dir, f))


# extract a single face from a given photograph
def extract_face(filename, aligned_dir, required_size=(160, 160)):
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
    image.save(aligned_dir)


# load images and extract faces for all images in a directory
def load_faces(directory, aligned_dir, required_size=(128, 128)):
    for filename in os.listdir(directory):
        path = os.path.join(directory, filename)
        save_path = os.path.join(aligned_dir, filename)
        if not os.path.exists(aligned_dir):
            os.mkdir(aligned_dir)
        extract_face(path, save_path, required_size=required_size)


ALIGNED_DIR = os.path.join(BASE_DIR, 'data', 'aligned')
if not os.path.exists(ALIGNED_DIR):
    os.mkdir(ALIGNED_DIR)

for dir in os.listdir(PICS_BASE):
    aligned_dir = os.path.join(BASE_DIR, 'data', 'aligned', dir)

    if not os.path.exists(aligned_dir):
        os.mkdir(aligned_dir)

    load_faces(os.path.join(BASE_DIR, 'data', 'test_pics', dir), aligned_dir)

for dir in os.listdir(ALIGNED_DIR):
    add_files(dir)

test_file = os.path.join(BASE_DIR, 'data', 'my_train.txt')
with open(test_file, 'wb') as file:
    for _ in range(5000):
        y = 0
        [file1, file2] = random.sample(list_pics, 2)
        base_1 = os.path.dirname(file1)
        base_2 = os.path.dirname(file2)

        if base_1 == base_2:
            y = 1

        pair = "{} {} {}\r\n".format(file1, file2, y)
        file.write(pair.encode())

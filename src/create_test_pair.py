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


def check_dir(file_name):
    if not os.path.exists(file_name):
        os.mkdir(file_name)


def add_files(dir):
    for f in os.listdir(dir):
        list_pics.append(os.path.join(dir, f))


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
        check_dir(aligned_dir)
        extract_face(path, save_path, required_size=required_size)


ALIGNED_DIR = os.path.join(BASE_DIR, 'data', 'aligned')
check_dir(ALIGNED_DIR)

ALIGNED_VAL_DIR = os.path.join(BASE_DIR, 'data', 'aligned_val')
check_dir(ALIGNED_VAL_DIR)

for dir in os.listdir(PICS_BASE):
    aligned_dir = os.path.join(BASE_DIR, 'data', 'aligned', dir)
    check_dir(aligned_dir)
    load_faces(os.path.join(BASE_DIR, 'data', 'test_pics', dir), aligned_dir)

for dir in os.listdir(ALIGNED_DIR):
    path = os.path.join(ALIGNED_DIR, dir)
    add_files(path)

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

list_pics = list()
for dir in os.listdir(os.path.join(BASE_DIR, 'data', 'val')):
    aligned_val_dir = os.path.join(BASE_DIR, 'data', 'aligned_val', dir)
    check_dir(aligned_val_dir)
    load_faces(os.path.join(BASE_DIR, 'data', 'val', dir), aligned_val_dir)

for dir in os.listdir(os.path.join(BASE_DIR, 'data', 'aligned_val')):
    path = os.path.join(BASE_DIR, 'data', 'aligned_val', dir)
    add_files(path)

test_file = os.path.join(BASE_DIR, 'data', 'my_val.txt')
with open(test_file, 'wb') as file:
    for _ in range(2000):
        y = 0
        [file1, file2] = random.sample(list_pics, 2)
        base_1 = os.path.dirname(file1)
        base_2 = os.path.dirname(file2)

        if base_1 == base_2:
            y = 1

        pair = "{} {} {}\r\n".format(file1, file2, y)
        file.write(pair.encode())

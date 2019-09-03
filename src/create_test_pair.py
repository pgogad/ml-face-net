import os
import random

BASE_DIR = os.path.dirname(__file__)
PICS_BASE = os.path.join(BASE_DIR, 'data', 'test_pics')

list_pics = list()


def add_files(dir):
    for f in os.listdir(os.path.join(PICS_BASE, dir)):
        list_pics.append(os.path.join(PICS_BASE, dir, f))


for dir in os.listdir(PICS_BASE):
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

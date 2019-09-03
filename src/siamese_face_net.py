import os

import keras.backend as K
import numpy as np
import tensorflow as tf

from fr_utils import load_weights_from_FaceNet, img_path_to_encoding
from inception_blocks_v2 import faceRecoModel

K.set_image_data_format('channels_first')
BASE_DIR = os.path.dirname(__file__)


def triplet_loss(y_true, y_pred, alpha=0.3):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    return loss


def prepare_val_database():
    database = dict()
    path = os.path.join(BASE_DIR, 'data', 'val')
    for d in os.listdir(path):
        f = os.path.join(path, d)
        for file in os.listdir(f):
            identity = d + '_' + os.path.splitext(os.path.basename(file))[0]
            database[identity] = os.path.join(f, file)
    return database


def prepare_database():
    database = dict()
    path = os.path.join(BASE_DIR, 'data', 'test_pics')
    for d in os.listdir(path):
        f = os.path.join(path, d)
        for file in os.listdir(f):
            identity = d + '_' + os.path.splitext(os.path.basename(file))[0]
            database[identity] = img_path_to_encoding(os.path.join(f, file), FRmodel)
    return database


def who_is_it(image, database, model):
    encoding = img_path_to_encoding(image, model)
    min_dist = 100
    identity = None
    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():
        dist = np.linalg.norm(db_enc - encoding)

        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.52:
        return None
    else:
        return identity


FRmodel = faceRecoModel(input_shape=(3, 96, 96))
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

data_set = prepare_database()
val_data_set = prepare_val_database()

total = 0
correct = 0

for k in val_data_set.keys():
    total += 1
    name = who_is_it(val_data_set[k], data_set, FRmodel)
    if name is not None:
        print('Expected name %s Actual name %s' % (k, name))
        if name[:3] == k[:3]:
            correct += 1
    else:
        print('Expected name %s Actual name %s' % (k, 'None'))

accuracy = (correct / total) * 100
print('Accuracy = %d ' % accuracy)

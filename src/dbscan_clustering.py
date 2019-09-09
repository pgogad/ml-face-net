from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import imageio
import numpy as np
import tensorflow as tf
from sklearn.cluster import DBSCAN

import detect_face
import facenet
import pickle

BASE_DIR = os.path.dirname(__file__)
MTCNN_DIR = os.path.join(BASE_DIR, 'data', 'model')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'model', 'facenet', '20180408-102900')
ALIGNED_DIR = os.path.join(BASE_DIR, 'data', 'lfw_aligned')
CLUSTER_DIR = os.path.join(BASE_DIR, 'data', 'clusters_dbscan')

if not os.path.exists(CLUSTER_DIR):
    os.mkdir(CLUSTER_DIR)


def create_network_face_detection(gpu_memory_fraction=0.5):
    with tf.compat.v1.Graph().as_default():
        gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.compat.v1.Session(config=tf.ConfigProto(gpu_options=gpu_option, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, MTCNN_DIR)
            return pnet, rnet, onet


def load_images_from_folder(folder):
    images = list()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if not os.path.isdir(path):
            continue
        for image in os.listdir(path):
            img = imageio.imread(os.path.join(path, image))
            if img is not None:
                images.append(img)
    return images


# model = KNeighborsClassifier(n_neighbors = 5)

def main():
    pnet, rnet, onet = create_network_face_detection()
    with tf.compat.v1.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            facenet.load_model(MODEL_DIR)

            images = load_images_from_folder(ALIGNED_DIR)
            images_placeholder = sess.graph.get_tensor_by_name("input:0")
            embeddings = sess.graph.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = sess.graph.get_tensor_by_name("phase_train:0")
            feed_dict = {images_placeholder: images, phase_train_placeholder: False}

            emb = sess.run(embeddings, feed_dict=feed_dict)
            nrof_images = len(images)
            matrix = np.zeros((nrof_images, nrof_images))

            for i in range(nrof_images):
                for j in range(nrof_images):
                    dist = np.sqrt(np.sum(np.square(np.subtract(emb[i, :], emb[j, :]))))
                    matrix[i][j] = dist

            with open('', 'wb') as file:
                pickle.dump(matrix, file)

            db = DBSCAN(eps=1.0, min_samples=0.3, metric='precomputed')
            db.fit(matrix)
            labels = db.labels_
            no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print('No of clusters:', no_clusters)

            print('Saving all clusters')
            for i in range(no_clusters):
                cnt = 1
                print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                path = os.path.join(CLUSTER_DIR, str(i))

                if not os.path.exists(path):
                    os.makedirs(path)

                    for j in np.nonzero(labels == i)[0]:
                        misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                        cnt += 1
                else:
                    for j in np.nonzero(labels == i)[0]:
                        misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                        cnt += 1


if __name__ == '__main__':
    main()

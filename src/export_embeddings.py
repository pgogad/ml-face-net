from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import time

import PIL.Image
import imageio
import numpy as np
import tensorflow as tf
from six.moves import xrange

import detect_face
import facenet


def main(args):
    train_set = facenet.get_dataset(args.data_dir)
    image_list, label_list = facenet.get_image_paths_and_labels(train_set)
    # fetch the classes (labels as strings) exactly as it's done in get_dataset
    path_exp = os.path.expanduser(args.data_dir)
    classes = [path for path in os.listdir(path_exp) if os.path.isdir(os.path.join(path_exp, path))]
    classes.sort()
    # get the label strings
    label_strings = [name for name in classes if os.path.isdir(os.path.join(path_exp, name))]

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            facenet.load_model(args.model_dir)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # Run forward pass to calculate embeddings
            nrof_images = len(image_list)
            print('Number of images: ', nrof_images)
            batch_size = args.image_batch
            if nrof_images % batch_size == 0:
                nrof_batches = nrof_images // batch_size
            else:
                nrof_batches = (nrof_images // batch_size) + 1
            print('Number of batches: ', nrof_batches)
            embedding_size = embeddings.get_shape()[1]
            emb_array = np.zeros((nrof_images, embedding_size))
            start_time = time.time()

            for i in range(nrof_batches):
                if i == nrof_batches - 1:
                    n = nrof_images
                else:
                    n = i * batch_size + batch_size
                # Get images for the batch
                if args.is_aligned is True:
                    images = facenet.load_data(image_list[i * batch_size:n], False, False, args.image_size)
                else:
                    images = load_and_align_data(image_list[i * batch_size:n], args.image_size, args.margin,
                                                 args.gpu_memory_fraction)
                feed_dict = {images_placeholder: images, phase_train_placeholder: False}
                # Use the facenet model to calcualte embeddings
                embed = sess.run(embeddings, feed_dict=feed_dict)
                emb_array[i * batch_size:n, :] = embed
                print('Completed batch', i + 1, 'of', nrof_batches)

            run_time = time.time() - start_time
            print('Run time: ', run_time)

            #   export emedings and labels
            label_list = np.array(label_list)

            np.save(args.embeddings_name, emb_array)
            np.save(args.labels_name, label_list)
            label_strings = np.array(label_strings)
            np.save(args.labels_strings_name, label_strings[label_list])


def load_and_align_data(image_paths, image_size, margin, gpu_memory_fraction):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        print(image_paths[i])
        # img = misc.imread(os.path.expanduser(image_paths[i]))
        img = imageio.imread(os.path.expanduser(image_paths[i]))
        img_size = np.asarray(img.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        det = np.squeeze(bounding_boxes[0, 0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
        bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
        cropped = img[bb[1]:bb[3], bb[0]:bb[2], :]
        # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        aligned = PIL.Image.fromarray(cropped).resize((image_size, image_size),
                                                      resample=PIL.Image.BILINEAR)
        prewhitened = facenet.prewhiten(aligned)
        img_list[i] = prewhitened
    images = np.stack(img_list)
    return images


BASE_DIR = os.path.dirname(__file__)
INPUT = os.path.join(BASE_DIR, 'data', 'lfw')
OUT_PUT = os.path.join(BASE_DIR, 'data', 'lfw_aligned')
MODEL_DIR = os.path.join(BASE_DIR, 'data', 'model', 'facenet', '20180408-102900')
EMBEDDING_DIR = os.path.join(BASE_DIR, 'data', 'model', 'facenet', 'embedding')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str,
                        help='Directory containing the meta_file and ckpt_file', default=MODEL_DIR)
    parser.add_argument('--data_dir', type=str,
                        help='Directory containing images. If images are not already aligned and cropped include --is_aligned False.',
                        default=OUT_PUT)
    parser.add_argument('--is_aligned', type=str,
                        help='Is the data directory already aligned and cropped?', default=True)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.',
                        default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.',
                        default=0.5)
    parser.add_argument('--image_batch', type=int,
                        help='Number of images stored in memory at a time. Default 500.',
                        default=500)

    #   numpy file Names
    parser.add_argument('--embeddings_name', type=str,
                        help='Enter string of which the embeddings numpy array is saved as.',
                        default=os.path.join(EMBEDDING_DIR, 'embeddings.npy'))
    parser.add_argument('--labels_name', type=str,
                        help='Enter string of which the labels numpy array is saved as.',
                        default=os.path.join(EMBEDDING_DIR, 'labels.npy'))
    parser.add_argument('--labels_strings_name', type=str,
                        help='Enter string of which the labels as strings numpy array is saved as.',
                        default=os.path.join(EMBEDDING_DIR, 'label_strings.npy'))
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

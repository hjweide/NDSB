#!/home/hendrik/work/ndsb/bin/python

import sys
import cv2
import numpy as np

from os import listdir
from os.path import join, dirname, basename


coarse_labels = ["acantharia", "amphipods", "appendicularian", "artifacts", "chaetognath", "chordate", "copepod", "crustacean", "ctenophore", "decapods", "detritus", "diatom", "echinoderm", "echinopluteus", "ephyra", "euphausiids", "fecal_pellet", "fish_larvae", "heteropod", "hydromedusae", "invertebrate", "jellies_tentacles", "polychaete", "protist", "pteropod", "radiolarian", "shrimp", "siphonophore", "stomatopod", "tornaria_acorn_worm_larvae", "trichodesmium", "trochophore_larvae", "tunicate", "unknown"]


def generate_training_data(allfile, datafile, labelsfile):
    images, labels = [], []
    with open(allfile, 'r') as ifile:
        for line in ifile:
            fname, label = line.strip().split(' ')
            img = cv2.imread(fname, cv2.CV_LOAD_IMAGE_GRAYSCALE)
            images.append(img.flatten())
            #labels.append(int(label))
            class_name = basename(dirname(fname))
            labels.append(class_name)

    images_array = np.vstack(images)
    labels_array = np.vstack(labels)

    print('np.shape(data) = %r' % (np.shape(images_array),))
    print('np.shape(labels) = %r' % (np.shape(labels_array),))

    # to resize the images back to their 2D-structure:
    # X = images_array.reshape(-1, 1, 48, 48)

    print('writing training data to %s...' % (datafile))
    with open(datafile, 'wb') as ofile:
        np.save(ofile, images_array)

    print('writing training labels to %s...' % (labelsfile))
    with open(labelsfile, 'wb') as ofile:
        np.save(ofile, labels_array)


def generate_test_data(indir, datafile, namesfile):
    allfiles = listdir(indir)
    images, names = [], []
    for fname in allfiles:
        img = cv2.imread(join(indir, fname), cv2.CV_LOAD_IMAGE_GRAYSCALE)
        images.append(img.flatten())
        names.append(fname)

    images_array = np.vstack(images)
    names_array = np.vstack(names)
    print('np.shape(data) = %r' % (np.shape(images_array),))
    print('np.shape(names) = %r' % (np.shape(names_array),))

    print('writing test data to %s' % (datafile))
    with open(datafile, 'wb') as ofile:
        np.save(ofile, images_array)

    print('writing filenames to %s...' % (namesfile))
    with open(namesfile, 'wb') as ofile:
        np.save(ofile, names_array)


if __name__ == '__main__':
    root = '/home/hendrik/work/ndsb/data'
    if len(sys.argv) != 2:
        print('usage:\n ./gen_data.py train\n ./gen_data.py test')
        exit(0)
    arg = sys.argv[1].lower()
    if arg == 'train':
        print('generating training data')
        allfile = join(root, 'kaggle-all-noaspect.txt')
        datafile = join(root, 'train_data_text.npy')
        labelsfile = join(root, 'train_labels_text.npy')

        generate_training_data(allfile, datafile, labelsfile)
    elif arg == 'test':
        print('generating test data')
        indir = join(root, 'test-resized-noaspect')
        datafile = join(root, 'test_data_noaspect.npy')
        namesfile = join(root, 'test_names_noaspect.npy')

        generate_test_data(indir, datafile, namesfile)

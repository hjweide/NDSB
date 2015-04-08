#!/home/hendrik/work/ndsb/bin/python

# file ensemble.py

from __future__ import absolute_import

import cv2
import sys
import time
import numpy as np
import cPickle as pickle

from sklearn.utils import shuffle
from sklearn.metrics import log_loss, accuracy_score  # NOQA
from sklearn.cross_validation import StratifiedKFold  # NOQA

from os.path import join, isfile
from net13 import TransformationBatchIterator  # NOQA
from net13 import AdjustVariable  # NOQA
from net13 import AdjustVariableOnStagnation  # NOQA
from net13 import EarlyStopping  # NOQA


big_params = {'max_offset': 6,
              'min_scale': 1 / 1.3,
              'max_scale': 1.3}

med_params = {'max_offset': 2,
              'min_scale': 1 / 1.3,
              'max_scale': 1.3}

small_params = {'max_offset': 2,
                'min_scale': 1 / 1.1,
                'max_scale': 1.1}

trans_params = {'max_offset': 8,
                'min_scale': 1 / 1.1,
                'max_scale': 1.1}


def perturb(Xi, theta, offset, scale):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        r00, r01, r10, r11 = scale * cos_theta, sin_theta, -sin_theta, scale * cos_theta

        x0, y0 = np.array(Xi.shape) / 2 + offset
        tx = x0 - r00 * x0 - r01 * y0
        ty = y0 - r10 * y0 - r11 * y0

        M = np.array([[r00, r01, tx],
                      [r10, r11, ty]], dtype=np.float32)

        return cv2.warpAffine(Xi, M, Xi.shape)


def generate_random_parameters(bs, params):
    max_offset = params['max_offset']
    max_scale = params['max_scale']
    min_scale = params['min_scale']
    radians = np.random.randint(360, size=bs) * (np.pi / 180)
    #radians = np.random.randint(12, size=bs) * 30 * (np.pi / 180)
    offsets = np.random.randint(2 * max_offset + 1, size=(2 * bs)).reshape(bs, 2) - max_offset
    scales = (max_scale - min_scale) * np.random.random(size=bs) + min_scale
    return radians, offsets, scales


def load(data_file, labels_file=None):
    data = np.load(data_file)

    data = 1. - (data.astype(np.float32) / 255.)

    if labels_file is not None:
        labels = np.load(labels_file)
        data, labels = shuffle(data, labels, random_state=42)
        labels = labels.flatten().astype(np.int32)
    else:
        labels = None

    return data, labels


def load2d(data_file, labels_file=None):
    data, labels = load(data_file, labels_file)
    data = data.reshape(-1, 1, 48, 48)

    return data, labels


def split_valid_train(data_file, labels_file, target='valid'):
    X, y = load2d(data_file, labels_file)
    if target == 'neither':
        return X, y

    kf = StratifiedKFold(y, round(1. / 0.2))
    if target == 'valid':
        _, indices = next(iter(kf))
    elif target == 'train':
        indices, _ = next(iter(kf))
    X_requested, y_requested = X[indices], y[indices]

    return X_requested, y_requested


# compute the geometric mean across the predictions from the different models in the ensemble
def compute_geometric_mean(mat_list, bias_low=True):
    if bias_low:
        log_mat_sum = np.sum(np.log(np.array(mat_list)), axis=0)
    else:
        log_mat_sum = np.sum(np.log(1. - np.array(mat_list)), axis=0)

    log_mat_sum /= len(mat_list)
    if bias_low:
        geom_mean = np.power(np.e, log_mat_sum)
    else:
        geom_mean = 1. - np.power(np.e, log_mat_sum)

    return geom_mean


# choose the most confidence prediction from the ensemble for each test image (worsened performance)
def pick_highest_confidence(mat_list):
    P = np.vstack([p.max(axis=1) for p in mat_list])
    probabilities = [x[i] for x, i in zip(zip(*mat_list), P.argmax(axis=0))]

    return np.vstack(probabilities)


def ensemble_models(net_file_list, aug_params_list, X, y=None, N=10):
    num_samples = X.shape[0]

    # store the predictions from each net
    ensemble_probabilities = []
    for net_file, aug_params in zip(net_file_list, aug_params_list):
        print('loading net %s from disk...' % (net_file))
        with open(net_file, 'rb') as ifile:
            net = pickle.load(ifile)

        # use the augmentation parameters to process each test image multiple times
        if aug_params is not None:
            # store the predictions from each transformation for this net
            net_probabilities = []
            for i in range(N):
                time_start = time.time()

                # perturb the data
                radians, offsets, scales = generate_random_parameters(num_samples, aug_params)
                Xc = np.copy(X)
                for ind, r, o, s in zip(range(num_samples), radians, offsets, scales):
                    Xc[ind, 0] = perturb(Xc[ind, 0], r, o, s)

                # now do predictions
                proba = net.predict_proba(Xc)
                net_probabilities.append(proba)
                time_end = time.time()
                print('  validation iteration %d of %d (%.1f s)' % (i + 1, N, time_end - time_start))

            # compute the geometric mean of all this net's predicted probabilities
            predicted_probs = compute_geometric_mean(net_probabilities, bias_low=True)
            #predicted_probs = pick_highest_confidence(net_probabilities)
        # process each test image only once
        else:
            predicted_probs = net.predict_proba(X)

        if y is not None:
            net_loss = log_loss(y, predicted_probs)
            net_acc = 100.0 * accuracy_score(y, [p.argmax() for p in predicted_probs])
            print('net %s, log loss = %.6f, accuracy = %.2f%%' % (net_file, net_loss, net_acc))
        ensemble_probabilities.append(predicted_probs)

    # compute the geometric mean of the probabilities from all nets
    ensemble_probabilities_mean = compute_geometric_mean(ensemble_probabilities, bias_low=True)
    #ensemble_probabilities_mean = pick_highest_confidence(ensemble_probabilities)

    return ensemble_probabilities_mean


def generate_submission(probabilities, names, submitfile, headerfile):
    print('generating kaggle submission...')
    # read the pre-generated header file
    with open(headerfile, 'r') as ifile:
        header = ifile.read()
    # write the predictions in csv format
    with open(submitfile, 'w') as ofile:
        ofile.write('%s\n' % (header))
        for fname, proba in zip(names, probabilities):
            ofile.write('%s,%s\n' % (fname[0], ','.join(['%.8f' % p for p in proba])))


if __name__ == '__main__':
    root = '/home/hendrik/work/ndsb/data'
    header_file = join(root, 'kaggle-header-noaspect.txt')
    #train_data_file = join(root, 'train_data_noaspect_coarse.npy')
    #train_labels_file = join(root, 'train_labels_noaspect_coarse.npy')
    train_data_file = join(root, 'train_data_noaspect.npy')
    train_labels_file = join(root, 'train_labels_noaspect.npy')
    test_data_file = join(root, 'test_data_noaspect.npy')
    test_names_file = join(root, 'test_names_noaspect.npy')
    submission_file = join(root, 'submissions', 'submit_10_13_19_21_24.csv')

    # the networks that will be used in the ensemble
    net_file_names = [
        'net10.pickle',
        'net13.pickle',
        'net19.pickle',
        'net21.pickle',
        'net24.pickle',
    ]

    # the augmentation parameters for test-time augmentation for each network
    aug_params_list = [
        big_params,
        big_params,
        small_params,
        None,
        med_params,
    ]

    assert len(net_file_names) == len(aug_params_list), 'need augmentation parameters for each network'

    net_file_list = [join(root, 'nets', nname) for nname in net_file_names]
    for fname in net_file_list:
        if not isfile(fname):
            print('could not find net file %s' % (fname))
            exit(1)
    if len(sys.argv) < 2:
        print('usage:\n ./ensemble.py valid\n ./ensemble.py test')
        exit(1)
    elif sys.argv[1].lower() == 'valid' or sys.argv[1].lower() == 'train':
        target = sys.argv[1].lower()
        print('train_data_file = %s' % (train_data_file))
        print('train_labels_file = %s' % (train_labels_file))
        data, labels = split_valid_train(train_data_file, train_labels_file, target=target)
        predicted_probs = ensemble_models(net_file_list, aug_params_list, data, labels, N=10)
        valid_ensemble_loss = log_loss(labels, predicted_probs)
        print('loss for ensemble of %d nets is %.6f' % (len(net_file_list), valid_ensemble_loss))
    elif sys.argv[1].lower() == 'test':
        print('test_data_file = %s' % (test_data_file))
        print('test_names_file = %s' % (test_names_file))
        print('header_file = %s' % (header_file))

        data, _ = load2d(test_data_file)
        names = np.load(test_names_file)

        predicted_probs = ensemble_models(net_file_list, aug_params_list, data, y=None, N=10)
        generate_submission(predicted_probs, names, submission_file, header_file)

#!/home/hendrik/work/ndsb/bin/python

# file visualize.py

from __future__ import absolute_import

import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from sklearn.utils import shuffle
from sklearn.metrics import log_loss  # NOQA
from sklearn.cross_validation import StratifiedKFold  # NOQA

from lasagne import layers  # NOQA
from lasagne.layers import cuda_convnet  # NOQA
from lasagne import nonlinearities  # NOQA
from nolearn.lasagne import NeuralNet  # NOQA
from nolearn.lasagne import BatchIterator  # NOQA

from os.path import join, basename, splitext

from net5 import TransformationBatchIterator  # NOQA
from net5 import AdjustVariable  # NOQA
from net5 import AdjustVariableOnStagnation  # NOQA
from net5 import EarlyStopping  # NOQA


def load(data_file, labels_file=None, names_file=None):
    data = np.load(data_file)

    data = 1. - (data.astype(np.float32) / 255.)

    if names_file is not None:
        with open(names_file, 'r') as ifile:
            names = np.array([basename(line.strip().split(' ')[0]) for line in ifile])
    else:
        names = None

    if labels_file is not None:
        labels = np.load(labels_file)
        data, labels, names = shuffle(data, labels, names, random_state=42)
        labels = labels.flatten().astype(np.int32)
    else:
        labels = None

    return data, labels, names


def load2d(data_file, labels_file=None, names_file=None):
    data, labels, names = load(data_file, labels_file, names_file)
    data = data.reshape(-1, 1, 48, 48)

    return data, labels, names


def load_training_data(data_file, labels_file, names_file):
    X, y, n = load2d(data_file, labels_file, names_file)
    kf = StratifiedKFold(y, round(1. / 0.2))
    train_indices, _ = next(iter(kf))
    X_train, y_train, names_train = X[train_indices], y[train_indices], n[train_indices]

    return X_train, y_train, names_train


def make_image_grid(img, name, examples_true, examples_false, names_true, names_false, label_true, label_pred, prob_true, prob_false, class_true, class_false, outfile):
    dim = 4
    fig = plt.figure(1)
    ax1 = plt.axes(frameon=False)
    # remove the outer frame axes
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    # need to raise the title a bit or it overlaps the labels
    plt.title('%s misclassified as %s\n%s vs. %s\n(p1 = %.5f, p2 = %.5f)' % (label_true, label_pred, class_true, class_false, prob_true, prob_false), y=1.08)
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(dim, dim),
                     axes_pad=0.5)
    for i in range(0, dim * dim):
        if i < 1:
            label = name
            sub_img = img
        elif (i < (2 * dim)):
            label = names_true[i - 1]
            sub_img = examples_true[i - 1, 0]
        else:
            label = names_false[i - (2 * dim)]
            sub_img = examples_false[i - (2 * dim), 0]

        grid[i].imshow(sub_img)
        grid[i].set_title(label=label)
        grid[i].get_xaxis().set_visible(False)
        grid[i].get_yaxis().set_visible(False)

    plt.savefig(outfile, bbox_inches='tight')


def visualize_misclassifications(data, labels_true, names, probs, classnames, outdir, N=4):
    num_samples = data.shape[0]
    for i, label_true, prob in zip(range(num_samples), labels_true, probs):
        label_pred = prob.argmax()
        if label_pred == label_true:
            continue
        name = names[i]
        class_true = classnames[label_true]
        class_false = classnames[label_pred]
        prob_true = prob[label_true]
        prob_false = prob.max()
        # get examples to give context to the image
        examples_true = data[labels_true == label_true]
        examples_false = data[labels_true == label_pred]
        names_true = names[labels_true == label_true]
        names_false = names[labels_true == label_pred]

        # don't want the same images each time
        rand_indices_true = np.random.choice(range(examples_true.shape[0]), size=(2 * N - 1), replace=False)
        rand_indices_false = np.random.choice(range(examples_false.shape[0]), size=(2 * N), replace=False)
        outfile = join(outdir, '%s.png' % splitext(name)[0])
        make_image_grid(data[i, 0], name, examples_true[rand_indices_true], examples_false[rand_indices_false], names_true[rand_indices_true], names_false[rand_indices_false], label_true, label_pred, prob_true, prob_false, class_true, class_false, outfile)


if __name__ == '__main__':
    root = '/home/hendrik/work/ndsb/data'
    outdir = join(root, 'misclassifications', 'net5')
    all_file = join(root, 'kaggle-all-noaspect.txt')
    header_file = join(root, 'kaggle-header-noaspect.txt')
    train_data_file = join(root, 'train_data_noaspect.npy')
    train_labels_file = join(root, 'train_labels_noaspect.npy')
    trained_net_file = join(root, 'nets', 'net5.pickle')

    print('train_data_file = %s' % (train_data_file))
    print('train_labels_file = %s' % (train_labels_file))
    print('header_file = %s' % (header_file))
    print('trained_net_file = %s' % (trained_net_file))

    data, labels, names = load_training_data(train_data_file, train_labels_file, all_file)
    with open(header_file, 'r') as ifile:
        header = ifile.read()
    classnames = header.strip().split(',')[1:]
    with open(trained_net_file, 'rb') as ifile:
        net = pickle.load(ifile)
    predicted_probabilities = net.predict_proba(data)
    visualize_misclassifications(data, labels, names, predicted_probabilities, classnames, outdir)

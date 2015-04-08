#!/home/hendrik/work/ndsb/bin/python

# this is net13.py but trained with fine-tuning for 10 additional epochs
# file net21.py

from __future__ import absolute_import

import cv2
import sys
import time
import theano   # NOQA
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import log_loss  # NOQA
from sklearn.cross_validation import StratifiedKFold  # NOQA

from lasagne import layers
from lasagne.layers import cuda_convnet
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator

from os.path import join

Conv2DLayer = cuda_convnet.Conv2DCCLayer
MaxPool2DLayer = cuda_convnet.MaxPool2DCCLayer


def float32(k):
    return np.cast['float32'](k)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        # changes the network's attribute matching the name (e.g., update_learning_rate)
        getattr(nn, self.name).set_value(new_value)


class AdjustVariableOnStagnation(object):
    def __init__(self, name, patience=100, update=lambda x: x):
        self.name = name
        self.patience = patience
        self.update = update
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.last_update_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.last_update_epoch = current_epoch
        elif self.last_update_epoch + self.patience < current_epoch:
            # when we update the values we need to reset the epoch count
            self.last_update_epoch = current_epoch
            current_value = getattr(nn, self.name).get_value()
            new_value = float32(self.update(current_value))
            # do not allow the learning rate to drop below 0 or the momentum go above 1 (should not happen!)
            if not (0 < new_value < 1):
                new_value = current_value
            getattr(nn, self.name).set_value(new_value)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['update']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()


max_offset = 6
min_scale, max_scale = 1. / 1.3, 1.3


class TransformationBatchIterator(BatchIterator):
    # apply a rotation of theta radians around the point defined by offset, and scale by scale
    def warp_image(self, Xi, theta, offset, scale):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        r00, r01, r10, r11 = scale * cos_theta, sin_theta, -sin_theta, scale * cos_theta

        x0, y0 = np.array(Xi.shape) / 2 + offset
        tx = x0 - r00 * x0 - r01 * y0
        ty = y0 - r10 * y0 - r11 * y0

        M = np.array([[r00, r01, tx],
                      [r10, r11, ty]], dtype=np.float32)

        return cv2.warpAffine(Xi, M, Xi.shape)

    def transform(self, Xb, yb):
        Xb, yb = super(TransformationBatchIterator, self).transform(Xb, yb)
        bs = Xb.shape[0]

        # copy the batch so that continuous changes don't corrupt the training set
        Xbc = np.copy(Xb)

        indices = np.array(range(bs))
        np.random.shuffle(indices)

        # generate random parameters for the transforms
        radians = np.random.randint(360, size=bs) * (np.pi / 180)
        offsets = np.random.randint(2 * max_offset + 1, size=(2 * bs)).reshape(bs, 2) - max_offset
        scales = np.e ** (np.log(max_scale) + (np.log(min_scale) - np.log(max_scale)) * np.random.random(size=bs))

        indices_transform = indices
        assert len(indices_transform) == len(radians) == len(offsets) == len(scales), 'lengths must match'
        for i, r, o, s in zip(indices_transform, radians, offsets, scales):
            Xbc[i, 0] = self.warp_image(Xbc[i, 0], r, o, s)

        return Xbc, yb

net21 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),

        ('conv1', Conv2DLayer),
        ('pool1', MaxPool2DLayer),
        ('maxout1', layers.FeaturePoolLayer),
        ('dropout1', layers.DropoutLayer),

        ('conv2', Conv2DLayer),
        ('pool2', MaxPool2DLayer),
        ('maxout2', layers.FeaturePoolLayer),
        ('dropout2', layers.DropoutLayer),

        ('conv3', Conv2DLayer),
        ('pool3', MaxPool2DLayer),
        ('maxout3', layers.FeaturePoolLayer),
        ('dropout3', layers.DropoutLayer),

        ('hidden5', layers.DenseLayer),
        ('maxout5', layers.FeaturePoolLayer),
        ('dropout5', layers.DropoutLayer),

        ('hidden6', layers.DenseLayer),
        ('maxout6', layers.FeaturePoolLayer),
        ('dropout6', layers.DropoutLayer),

        ('output', layers.DenseLayer),
    ],
    input_shape=(None, 1, 48, 48),

    conv1_num_filters=256, conv1_filter_size=(4, 4), conv1_strides=(1, 1),
    pool1_ds=(2, 2), pool1_strides=(2, 2),
    maxout1_ds=2,
    dropout1_p=0.1,

    conv2_num_filters=384, conv2_filter_size=(3, 3), conv2_strides=(1, 1),
    pool2_ds=(2, 2), pool2_strides=(2, 2),
    maxout2_ds=2,
    dropout2_p=0.2,

    conv3_num_filters=512, conv3_filter_size=(3, 3), conv3_strides=(1, 1),
    pool3_ds=(2, 2), pool3_strides=(2, 2),
    maxout3_ds=2,
    dropout3_p=0.3,

    hidden5_num_units=8192,
    maxout5_ds=2,
    dropout5_p=0.5,

    hidden6_num_units=8192,
    maxout6_ds=2,
    dropout6_p=0.5,

    output_num_units=121,
    output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),
    on_epoch_finished=[
        AdjustVariableOnStagnation('update_learning_rate', patience=25, update=lambda x: 0.5 * x),
        #AdjustVariableOnStagnation('update_momentum', patience=25, update=lambda x: (1 - x) * 0.9 + x),
        #AdjustVariable('update_learning_rate', start=0.03, stop=0.001),
        #AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=50)
    ],
    #batch_iterator_train=TransformationBatchIterator(batch_size=128),

    max_epochs=250,
    verbose=1,
)


def load(data_file, labels_file=None):
    data = np.load(data_file)

    data = 1. - (data.astype(np.float32) / 255.)

    #mean = np.mean(data, axis=0)
    #data -= mean

    #std = np.std(data, axis=0)
    #data /= std

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


def train_net(X, y, net, netfile=None, weights_file=None):
    print('np.shape(X) = %r' % (np.shape(X),))
    print('np.shape(y) = %r' % (np.shape(y),))
    print('netfile = %s' % (netfile))

    if weights_file is not None:
        with open(weights_file, 'rb') as ifile:
            net_pretrain = np.load(ifile)
            net.load_weights_from(net_pretrain)

    net.fit(X, y)

    if netfile is not None:
        with open(netfile, 'wb') as pfile:
            pickle.dump(net, pfile, protocol=pickle.HIGHEST_PROTOCOL)


def plot_loss(net, outfile=None):
    train_loss = np.array([i['train_loss'] for i in net.train_history_])
    valid_loss = np.array([i['valid_loss'] for i in net.train_history_])

    plt.plot(train_loss, 'b', linewidth=3, label='train')
    plt.plot(valid_loss, 'g', linewidth=3, label='valid')

    plt.grid()
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.yscale('log')

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
    plt.show()


def perturb(Xi, theta, offset, scale):
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        r00, r01, r10, r11 = scale * cos_theta, scale * sin_theta, - scale * sin_theta, scale * cos_theta

        x0, y0 = np.array(Xi.shape) / 2 + offset
        tx = x0 - r00 * x0 - r01 * y0
        ty = y0 - r10 * y0 - r11 * y0

        M = np.array([[r00, r01, tx],
                      [r10, r11, ty]], dtype=np.float32)

        return cv2.warpAffine(Xi, M, Xi.shape)


def generate_random_parameters(bs):
        radians = np.random.randint(360, size=bs) * (np.pi / 180)
        offsets = np.random.randint(2 * max_offset + 1, size=(2 * bs)).reshape(bs, 2) - max_offset
        scales = (max_scale - min_scale) * np.random.random(size=bs) + min_scale
        return radians, offsets, scales


def generate_submission(X, names, net, submitfile, headerfile, N=10):
    # read the pre-generated header file
    with open(headerfile, 'r') as ifile:
        header = ifile.read()

    bs = X.shape[0]
    probabilities = np.zeros((bs, 121), dtype=np.float32)
    for i in range(N):
        time_start = time.time()
        radians, offsets, scales = generate_random_parameters(bs)
        Xc = np.copy(X)
        for ind, r, o, s in zip(range(bs), radians, offsets, scales):
            Xc[ind, 0] = perturb(Xc[ind, 0], r, o, s)
        proba = net.predict_proba(Xc)
        probabilities += proba
        time_end = time.time()
        print('validation iteration %d of %d (%.3f s)' % (i + 1, N, time_end - time_start))

    # get the actual predictions
    y = probabilities / N

    # write the predictions in csv format
    with open(submitfile, 'w') as ofile:
        ofile.write('%s\n' % (header))
        for fname, prediction in zip(names, y):
            ofile.write('%s,%s\n' % (fname[0], ','.join(['%.8f' % p for p in prediction])))


if __name__ == '__main__':
    root = '/home/hendrik/work/ndsb/data'
    header_file = join(root, 'kaggle-header-noaspect.txt')
    train_data_file = join(root, 'train_data_noaspect.npy')
    train_labels_file = join(root, 'train_labels_noaspect.npy')
    test_data_file = join(root, 'test_data_noaspect.npy')
    test_names_file = join(root, 'test_names_noaspect.npy')
    trained_net_file = join(root, 'nets', 'net21.pickle')
    pretrained_net_file = join(root, 'nets', 'net13.pickle')
    submission_file = join(root, 'submissions', 'submit_11.csv')

    retrain, test = False, False
    if len(sys.argv) < 2:
        print('usage:\n ./net21.py test\n ./net21.py retrain')
    elif sys.argv[1].lower() == 'retrain':
        retrain = True
        print('retrain = %s' % (retrain))
        print('train_data_file = %s' % (train_data_file))
        print('train_labels_file = %s' % (train_labels_file))
    elif sys.argv[1].lower() == 'test':
        test = True
        print('test = %s' % (test))
        print('test_data_file = %s' % (test_data_file))
        print('test_names_file = %s' % (test_names_file))
        print('header_file = %s' % (header_file))

    print('trained_net_file = %s' % (trained_net_file))

    if retrain:
        print('retraining net...')
        data, labels = load2d(train_data_file, train_labels_file)
        train_net(data, labels, net21, trained_net_file, weights_file=pretrained_net_file)
        plot_loss(net21, outfile=join(root, 'plots', 'net21_loss.png'))
    else:
        print('loading net from disk...')
        with open(trained_net_file, 'rb') as ifile:
            net21 = pickle.load(ifile)
        print('loading test data...')
        data, _ = load2d(test_data_file)
        names = np.load(test_names_file)
        print('generating kaggle submission...')

        generate_submission(data, names, net21, submission_file, header_file, N=50)

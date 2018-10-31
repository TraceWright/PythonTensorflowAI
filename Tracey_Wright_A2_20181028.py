'''IFN680 Assignment 2
Semester 2 2018
Tracey Wright n9131302
'''

# activate tensorflow on cmd or terminal prior to running code

import numpy as np
import math
import time
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
import random

from keras.datasets import mnist
from keras.models import Model
from keras.callbacks import Callback
from keras.layers import Input, Flatten, Dense, Dropout, Conv2D, MaxPooling2D, Lambda
from keras.optimizers import RMSprop
from keras import backend as K
import tensorflow as tf

epochs = 2

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        acc = self.times[len(self.times) - 1] if len(self.times) > 0 else 0
        self.times.append(round(time.time() - self.epoch_time_start + acc, 2))

# create csv file prior to calling this function
def report_loss_over_time(loss, val_loss, mean_pred, val_mean_pred, times, full_path):

    with open(full_path, 'a') as fp:
        fp.write("Time, Loss, Validation Loss" + "\n")
        for line in zip(times, loss, val_loss, mean_pred, val_mean_pred):
            fp.write(",".join(str(v) for v in line) + "\n")

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))

    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)

def create_pairs(x, digit_indices, num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)

def create_base_network(input_shape, num_classes):
    '''Base network to be shared (eq. to feature extraction).
    '''

    # layer configuration for initial training run
    # input = Input(shape=input_shape)
    # x = Conv2D(32, kernel_size=(4, 4),
    #                  activation='relu',
    #                  input_shape=input_shape)(input)
    # x = Conv2D(64, (3, 3), activation='relu')(x)
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.25)(x)
    # x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(num_classes, activation='softmax')(x)

    # return Model(input, x)


    # faster running model - marginally greater precision and recall
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)


def compute_accuracy(y_true, y_pred, accuracy_threshold=0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < accuracy_threshold
    return np.mean(pred == y_true)


def mean_pred(y_true, y_pred, accuracy_threshold=0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < accuracy_threshold, y_true.dtype)))

def split_indices(train_split, digit_indices):
    '''split indices into training (%) and test (%)
    '''
    train_digit_indices =[]
    test_digit_indices =[]
    
    for i in range(len(digit_indices_inner)):
        np.random.shuffle(digit_indices_inner[i])
        arr = digit_indices_inner[i]
        len_di = len(digit_indices_inner[i])
        ln = math.floor(len_di * train_split)
        train_digit_indices.append(arr[0:ln])
        test_digit_indices.append(arr[ln:len_di])
    
    return train_digit_indices, test_digit_indices

# create csv file prior to calling this function
def classification_report_to_csv(ground_truth,
                                 predictions,
                                 accuracy,
                                 full_path,
                                 test_name):

    labels = unique_labels(ground_truth, predictions)
    precision, recall, f_score, support = precision_recall_fscore_support(ground_truth,
                                                                          predictions,
                                                                          labels=labels,
                                                                          average=None)
    named_labels = [test_name + " - " + str(labels[0]), test_name]
    print(full_path, labels, precision, recall, accuracy)

    with open(full_path, 'a') as fp:
        # fp.write(test_name + "," + accuracy + "," + "\n")
        for line in zip(named_labels, precision, recall):
            fp.write(",".join(str(v) for v in line) + "\n")
        fp.write("\n")

# test contrastive loss function
class TestAssignmentTwo(tf.test.TestCase):
    def test_contrastive_loss(self):
        with self.test_session():
            for x in range(10):
                y_tr = random.randint(1, 10)
                y_pr = random.randint(1, 10)
                x = contrastive_loss(y_tr, y_pr)
                self.assertEqual(x.eval(), y_tr*pow(y_pr,2))

if __name__ == "__main__":

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # pool training and test data 
    x_data = np.concatenate((x_train, x_test), axis=0)
    y_data = np.concatenate((y_train, y_test), axis=0)
    data_range = 255

    # can comment out this line once saved locally
    np.savez('mnist_dataset.npz', x_data = x_data, y_data = y_data)

    with np.load('mnist_dataset.npz') as npzfile:
            x_data = npzfile['x_data']
            y_data = npzfile['y_data']

    x_data = x_data.astype('float32')
    x_data /= data_range

    # reshape the input arrays to 4D (batch_size, rows, columns, channels)
    img_rows, img_cols = x_data.shape[1:3]
    x_data = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    accuracy_threshold = 0.5
    num_classes = len(np.unique(y_train))
    train_split = 0.8

    digits_inner = [2,3,4,5,6,7]
    digits_outer = [0,1,8,9]
    digits_all = [0,1,2,3,4,5,6,7,8,9]

    digit_indices_inner = [np.where(y_data == i)[0] for i in digits_inner]
    digit_indices_outer = [np.where(y_data == i)[0] for i in digits_outer]
    digit_indices_all = [np.where(y_data == i)[0] for i in digits_all]

    train_digit_indices, test_digit_indices_case1 = split_indices(train_split, digit_indices_inner)

    # create_pairs
    tr_pairs, tr_y = create_pairs(x_data, train_digit_indices, len(digits_inner))
    te1_pairs, te1_y = create_pairs(x_data, test_digit_indices_case1, len(digits_inner))
    te2_pairs, te2_y = create_pairs(x_data, digit_indices_all, len(digits_all))
    te3_pairs, te3_y = create_pairs(x_data, digit_indices_outer, len(digits_outer))

    # network definition
    base_network = create_base_network(input_shape, num_classes)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # the weights of the network are shared across the two branches
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)

    distance = Lambda(euclidean_distance,
                    output_shape=eucl_dist_output_shape)([processed_a, processed_b])

    model = Model([input_a, input_b], distance)
    time_callback = TimeHistory()

    # train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=['accuracy', mean_pred])
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
            batch_size=128,
            epochs=epochs,
            validation_data=([te1_pairs[:, 0], te1_pairs[:, 1]], te1_y),
            callbacks=[time_callback])

    times = time_callback.times
    # print(history.history, times)

    # report_loss_over_time(history.history['loss'], history.history['val_loss'], history.history['mean_pred'], history.history['val_mean_pred'], times, "loss_over_time.csv")

    # compute final accuracy on training and test sets
    y_pred_tr = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(tr_y, y_pred_tr, accuracy_threshold)

    y_pred = model.predict([te1_pairs[:, 0], te1_pairs[:, 1]])
    te_acc = compute_accuracy(te1_y, y_pred, accuracy_threshold)

    y_pred2 = model.predict([te2_pairs[:, 0], te2_pairs[:, 1]])
    te_acc2 = compute_accuracy(te2_y, y_pred2, accuracy_threshold)

    y_pred3 = model.predict([te3_pairs[:, 0], te3_pairs[:, 1]])
    te_acc3 = compute_accuracy(te3_y, y_pred3, accuracy_threshold)

    # report test metrics
    # classification_report_to_csv(tr_y, y_pred_tr.ravel() < accuracy_threshold, str(tr_acc), "test_accuracy.csv", "training")
    # classification_report_to_csv(te1_y, y_pred.ravel() < accuracy_threshold, str(te_acc), "test_accuracy.csv", "test_234567")
    # classification_report_to_csv(te2_y, y_pred2.ravel() < accuracy_threshold, str(te_acc2), "test_accuracy.csv", "test_0123456789")
    # classification_report_to_csv(te3_y, y_pred3.ravel() < accuracy_threshold, str(te_acc3), "test_accuracy.csv", "test_0189")


    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc2))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc3))

    tf.test.main()

    # adapted from: https://gist.github.com/mmmikael/0a3d4fae965bdbec1f9d
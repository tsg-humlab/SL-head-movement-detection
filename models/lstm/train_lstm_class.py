import argparse
import pickle
from pathlib import Path

import numpy as np
import torch 
import random

from models.processing.preparation import separate_seqs, remove_nan_values_from_tensors, separate_seqs_all, separate_seqs_all_with_return_seq, plot_pyr_graphs, shuffle_data
from models.processing.facial_movement import derivatives_from_csv
from utils.frames_csv import load_df, load_all_labels
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.utils import class_weight

import matplotlib.pyplot as plt
import math

MINIMUM_SEQUENCE_LENGTH = 2
MODEL_LSTM_CLASS_FILENAME = 'train_lstm_class.h5'
MASK_VALUE = [-10]*4
PAD_VALUE = [0]*4

import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers.legacy import SGD
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Masking, Bidirectional, Dropout, Input, BatchNormalization
from keras.metrics import Precision, Recall
from keras.regularizers import l2
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

def augment_flip_data(data_x, data_y):
    """
    double the data by adding horizontally flipped modements
    """
    new_data_x, new_data_y = [], []
    for d_i, d_x in enumerate(data_x):
        # Add original data
        new_data_x.append(d_x)
        new_data_y.append(data_y[d_i])

        # Add flipped data
        new_data_x_video = []
        for d_x_frame in d_x:
            new_frame = d_x_frame.copy()
            new_frame[1] *= -1
            new_frame[2] *= -1
            new_frame[3] *= -1
            new_data_x_video.append(new_frame)
        new_data_x.append(new_data_x_video)
        new_data_y.append(data_y[d_i])
    
    return (new_data_x, new_data_y)

def plot_pyr_graphs(vector, number=2, diff_pix=False, diff_ang=False, movement_type='Movement', ang_norm=False):
    import random
    randoms = random.sample(range(0, len(vector)), number)
    for ra in randoms:
        p, y, r, s = [], [], [], []
        for xi, xx in enumerate(vector[ra]):
            if not diff_pix:
                p.append(xx[0])
                y.append(xx[1])
                r.append(xx[2])
                s.append(xx[3])
            if diff_pix:
                if xi == 0:
                    p.append(0)
                    y.append(0)
                    r.append(0)
                    s.append(0)
                p.append(xx[0])
                y.append(xx[1])
                r.append(xx[2])
                s.append(xx[3])
            if xi != 0 and diff_pix:
                p[-1] = p[-1]+p[-2]
                y[-1] = y[-1]+y[-2]
                r[-1] = r[-1]+r[-2]
                s[-1] = s[-1]+s[-2]
        from matplotlib import pyplot as plt
        plt.figsize=(30,6)
        plt.plot(p, label='pitch')
        plt.plot(y, label='yaw')
        plt.plot(r, label='roll')
        plt.plot(s, label='shoulder')
        plt.ylabel('pixels')
        plt.xlabel('frame')
        plt.title(movement_type + " pixels")
        plt.legend()
        plt.show()

        p, y, r, s = [], [], [], []
        for xi, xx in enumerate(vector[ra]):
            if not diff_ang:
                if ang_norm:
                    p.append((xx[4]-0.5)*2*90)
                    y.append((xx[5]-0.5)*2*90)
                    r.append((xx[6]-0.5)*2*90)
                    s.append((xx[7]-0.5)*2*90)
                else:
                    p.append(xx[4])
                    y.append(xx[5])
                    r.append(xx[6])
                    s.append(xx[7])
            if diff_ang:
                if xi == 0:
                    p.append(0)
                    y.append(0)
                    r.append(0)
                    s.append(0)
                p.append(xx[4])
                y.append(xx[5])
                r.append(xx[6])
                s.append(xx[7])
            if xi != 0 and diff_ang:
                p[-1] = p[-1]+p[-2]
                y[-1] = y[-1]+y[-2]
                r[-1] = r[-1]+r[-2]
                s[-1] = s[-1]+s[-2]
        from matplotlib import pyplot as plt
        plt.figsize=(30,6)
        plt.plot(p, label='pitch')
        plt.plot(y, label='yaw')
        plt.plot(r, label='roll')
        plt.plot(s, label='shoulder')
        plt.ylabel('degrees')
        plt.xlabel('frame')
        plt.title(movement_type + " angles")
        plt.legend()
        plt.show()


def fit_lstms_class(df_train, df_val, model_dir, overwrite=True, debug=False, epochs=100, window_size=36):
    if ((Path(model_dir)/MODEL_LSTM_CLASS_FILENAME).exists()) and not overwrite:
        if not debug:
            raise FileExistsError('The models already exist in this directory, you can explicitly allow overwriting')

    print('Retrieving the training labels')
    train_y = load_all_labels(df_train, shift=1)
    print('Calculating training vector derivatives')
    train_x = derivatives_from_csv(df_train)
    nr_features = len(train_x[0][0])
    print('Separating training sequences')
    train_background_vectors, train_shake_vectors, train_nod_vectors = separate_seqs(train_x, train_y, MINIMUM_SEQUENCE_LENGTH)
    
    plot_pyr_graphs(train_shake_vectors, number=1, diff_pix=False, diff_ang=False, movement_type='Shake')
    plot_pyr_graphs(train_nod_vectors, number=1, diff_pix=False, diff_ang=False, movement_type='Nod')

    print('Retrieving the validation labels')
    val_y = load_all_labels(df_val, shift=1)
    print('Calculating validation vector derivatives')
    val_x = derivatives_from_csv(df_val)
    print('Separating validation sequences')
    val_background_vectors, val_shake_vectors, val_nod_vectors = separate_seqs(val_x, val_y, MINIMUM_SEQUENCE_LENGTH)
    
    # print('Augmenting data')
    train_nod_vectors, _ = augment_flip_data(train_nod_vectors, [2]*len(train_nod_vectors))
    val_nod_vectors, _ = augment_flip_data(val_nod_vectors, [2]*len(val_nod_vectors))

    print('Check number of data points: ')
    min_train = len(train_nod_vectors)
    min_val = len(val_nod_vectors)
    print("Min train: ", min_train, ", min val: ", min_val)

    # print('Take a sample randomly: disable this when class weights are used')
    # train_background_vectors = random.sample(train_background_vectors, min_train)
    # train_shake_vectors = random.sample(train_shake_vectors, min_train)
    # train_nod_vectors = random.sample(train_nod_vectors, min_train)
    # val_background_vectors = random.sample(val_background_vectors, min_val)
    # val_shake_vectors = random.sample(val_shake_vectors, min_val)
    # val_nod_vectors = random.sample(val_nod_vectors, min_val)

    # print('Make y values')
    train_background_y = [0]*len(train_background_vectors)
    train_shake_y = [1]*len(train_shake_vectors)
    train_nod_y = [2]*len(train_nod_vectors)
    val_background_y = [0]*len(val_background_vectors)
    val_shake_y = [1]*len(val_shake_vectors)
    val_nod_y = [2]*len(val_nod_vectors)

    print('Concatenate to form 1 list')
    train_x = np.array(train_background_vectors + train_shake_vectors + train_nod_vectors)
    train_y = np.array(train_background_y + train_shake_y + train_nod_y)
    val_x = np.array(val_background_vectors + val_shake_vectors + val_nod_vectors)
    val_y = np.array(val_background_y + val_shake_y + val_nod_y)

    print('Scaling data')
    from sklearn.preprocessing import StandardScaler
    scalers = {}
    for i in range(train_x.shape[2]):
        scalers[i] = StandardScaler()
        train_x[:, :, i] = scalers[i].fit_transform(train_x[:, :, i].reshape(-1, 1)).reshape(train_x[:, :, i].shape)

    for i in range(val_x.shape[2]):
        val_x[:, :, i] = scalers[i].transform(val_x[:, :, i].reshape(-1, 1)).reshape(val_x[:, :, i].shape)

    # print('Shuffle')
    c = list(zip(train_x, train_y))
    random.shuffle(c)
    train_x, train_y = zip(*c)
    c = list(zip(val_x, val_y))
    random.shuffle(c)
    val_x, val_y = zip(*c)

    print('Convert numpy train arrays to lists')
    new_train_x, new_val_x = [], []
    for x_t in train_x:
        new_list = []
        for x in x_t:
            new_list.append(list(x))
        new_train_x.append(list(new_list))
    train_x = new_train_x
    for x_t in val_x:
        new_list = []
        for x in x_t:
            new_list.append(list(x))
        new_val_x.append(list(new_list))
    val_x = new_val_x

    # print('Histogram for lengths of shakes')
    # lengths = []
    # for x_i, x_t in enumerate(train_x):
    #     if len(x_t)<100 and train_y[x_i] not in [0,2]:
    #         lengths.append(len(x_t))
    # for x_i, x_t in enumerate(val_x):
    #     if len(x_t)<100 and val_y[x_i] not in [0,2]:
    #         lengths.append(len(x_t))
    # plt.hist(lengths, bins=100)
    # plt.show() 

    # print('Histogram for lengths of nods')
    # lengths = []
    # for x_i, x_t in enumerate(train_x):
    #     if len(x_t)<100 and train_y[x_i] not in [0,1]:
    #         lengths.append(len(x_t))
    # for x_i, x_t in enumerate(val_x):
    #     if len(x_t)<100 and val_y[x_i] not in [0,1]:
    #         lengths.append(len(x_t))
    # plt.hist(lengths, bins=100)
    # plt.show() 
    
    # Pad with [-10,-10,-10,-10] to get to length
    print('Pad sequences')
    for x_i, x_t in enumerate(train_x):
        if len(x_t)>window_size:
            train_x[x_i] = x_t[:window_size]
        else:
            # place array in the middle of a padded array
            padded_arr = [PAD_VALUE]*window_size
            # Divide by 2 and round down, e.g. (9-4)/5 -> 2
            start = math.floor((window_size - len(x_t)) / 2)
            padded_arr[start:start + len(x_t)] = x_t
            # padded_arr[:start] = [x_t[0]] * start
            padded_arr[start + len(x_t) : window_size] = [padded_arr[start + len(x_t) - 1]] * (window_size - len(x_t) - start)
            train_x[x_i] = padded_arr

    for x_i, x_t in enumerate(val_x):
        if len(x_t)>window_size:
            val_x[x_i] = x_t[:window_size]
        else:
            # place array in the middle of a padded array
            padded_arr = [PAD_VALUE]*window_size
            # Divide by 2 and round down, e.g. (9-4)/5 -> 2
            start = math.floor((window_size - len(x_t)) / 2)
            padded_arr[start:start + len(x_t)] = x_t
            # padded_arr[:start] = [x_t[0]] * start
            padded_arr[start + len(x_t) : window_size] = [padded_arr[start + len(x_t) - 1]] * (window_size - len(x_t) - start)
            val_x[x_i] = padded_arr

    plot_pyr_graphs(train_x, number=2, diff_pix=False, diff_ang=False, movement_type='Unk')

    print('Make dictionaries')
    simple_sign_dict = {'background': 0, 'shake': 1, 'nod': 2}
    turned_simple_sign_dict = {}
    for ryi, ry in enumerate(list(set(simple_sign_dict))):
        turned_simple_sign_dict[ryi] = ry
    # print(turned_simple_sign_dict)

    # print('Convert to categorical')
    # train_y = to_categorical(train_y)
    # val_y = to_categorical(val_y)

    # print('Convert to numpy arrays')
    # train_x = np.array(train_x)
    # train_y = np.array(train_y)
    # val_x = np.array(val_x)
    # val_y = np.array(val_y)

    nr_classes = len(simple_sign_dict)
    print("Nr of classes = ", nr_classes)

    callbacks1 = ModelCheckpoint('data/train_lstm_class.hdf5', save_best_only=True)
    callbacks2 = EarlyStopping(
        monitor='val_loss', 
        patience = 10, 
        verbose = 1,
        restore_best_weights=True)

    callbacks = [callbacks1,callbacks2]

    print('Creating model')
    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(window_size, nr_features)))
    model.add(Bidirectional(LSTM(64,return_sequences=True, kernel_regularizer=l2(0.001))))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(34)))
    # model.add(LSTM(64,return_sequences=True, activation='sigmoid'))
    # model.add(LSTM(34,activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(nr_classes, activation = 'softmax'))

    # optimizer = Adam(learning_rate=0.005)
    # lr_schedule = ExponentialDecay(
    # initial_learning_rate=0.05,
    # decay_steps=10000,
    # decay_rate=0.9)
    # optimizer = SGD(learning_rate=lr_schedule)
    optimizer = Adam(learning_rate=0.001)  # Adjust as needed
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy", Precision(), Recall()])

    # Add class weights for unbalanced training
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(train_y), y=train_y)
    class_weights = dict(enumerate(class_weights))
    print("Class weights: ", class_weights)
    
    # train_x = train_x.tolist()
    # val_x = val_x.tolist()
    # train_y = train_y.tolist()  
    # val_y = val_y.tolist()

    history = model.fit(train_x, train_y, validation_data=[val_x, val_y], epochs=10, callbacks=callbacks, batch_size=32, class_weight=class_weights)

    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    Y_pred = np.argmax(model.predict(val_x),axis=1)
    output = np.argmax(val_y, axis=1)

    from sklearn.metrics import accuracy_score
    print("Accuracy: ", accuracy_score(output, Y_pred))

    actual = output
    predicted = Y_pred

    confusion_matrix = metrics.confusion_matrix(actual, predicted)
    from validation.cross_validation import plot_confusion_matrix
    plot_confusion_matrix(confusion_matrix, title=None, labels = simple_sign_dict)

    print("Saving model")
    model.save(model_dir/MODEL_LSTM_CLASS_FILENAME)

    print('Training program complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    fit_lstms_class(args.frames_csv, args.model_dir, args.overwrite)

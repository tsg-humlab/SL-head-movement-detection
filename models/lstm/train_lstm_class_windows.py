import argparse
import pickle
from pathlib import Path

import numpy as np
import torch 
import random
import joblib
import matplotlib.pyplot as plt
import math

from models.processing.preparation import separate_seqs_all_with_return_seq, plot_pyr_graphs, shuffle_data
from models.processing.facial_movement import derivatives_from_csv
from utils.frames_csv import load_df, load_all_labels
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from validation.validate_model import plot_confusion_matrix
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

from models.lstm.test_lstm_class_windows import MODEL_LSTM_CLASS_FILENAME

MASK_VALUE = [-10]*4

import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers.legacy import SGD
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Masking, Bidirectional, Dropout, Input, Attention, concatenate, BatchNormalization
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.regularizers import l2
from keras.metrics import AUC

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def balance_data(train_x, train_y, val_x, val_y, window_size):
    # Creating a balanced training dataset
    train_y_middles = train_y[:, math.floor(window_size / 2)]

    # random sample of indexes based on class label
    train_x_bg = train_x[train_y_middles == 0]
    train_y_bg = train_y[train_y_middles == 0]
    train_x_sh = train_x[train_y_middles == 1]
    train_y_sh = train_y[train_y_middles == 1]
    train_x_nod = train_x[train_y_middles == 2]
    train_y_nod = train_y[train_y_middles == 2]
    least_nr = min(len(train_x_bg), len(train_x_sh), len(train_x_nod))

    # Randomly sample from the background, shake, and nod sequences keeping x together with y
    train_x_bg, train_y_bg = shuffle_data(train_x_bg, train_y_bg, nr_samples=least_nr) 
    train_x_sh, train_y_sh = shuffle_data(train_x_sh, train_y_sh, nr_samples=least_nr)
    train_x_nod, train_y_nod = shuffle_data(train_x_nod, train_y_nod, nr_samples=least_nr)
    
    # Concatenate the background, shake, and nod sequences
    train_x = np.concatenate((train_x_bg, train_x_sh, train_x_nod))
    train_y = np.concatenate((train_y_bg, train_y_sh, train_y_nod))

    # Creating a balanced validation dataset
    val_y_middles = val_y[:, math.floor(window_size / 2)]
    # random sample of indexes based on class label
    val_x_bg = val_x[val_y_middles == 0]
    val_y_bg = val_y[val_y_middles == 0]
    val_x_sh = val_x[val_y_middles == 1]
    val_y_sh = val_y[val_y_middles == 1]
    val_x_nod = val_x[val_y_middles == 2]
    val_y_nod = val_y[val_y_middles == 2]
    least_nr = min(len(val_x_bg), len(val_x_sh), len(val_x_nod))

    # Randomly sample from the background, shake, and nod sequences keeping x together with y
    val_x_bg, val_y_bg = shuffle_data(val_x_bg, val_y_bg, nr_samples=least_nr)
    val_x_sh, val_y_sh = shuffle_data(val_x_sh, val_y_sh, nr_samples=least_nr)
    val_x_nod, val_y_nod = shuffle_data(val_x_nod, val_y_nod, nr_samples=least_nr)

    # Concatenate the background, shake, and nod sequences
    val_x = np.concatenate((val_x_bg, val_x_sh, val_x_nod))
    val_y = np.concatenate((val_y_bg, val_y_sh, val_y_nod))

    return train_x, train_y, val_x, val_y

def fit_lstms_class_windows(df_train, df_val, model_dir, window_size=37, load_values=False):

    if load_values and (Path(model_dir)/'train_x_windows.npy').exists() and (Path(model_dir)/'val_x_windows.npy').exists()\
                   and (Path(model_dir)/'train_y_windows.npy').exists() and (Path(model_dir)/'val_y_windows.npy').exists():
        print('Loading train and val data from numpy')
        train_x, train_y, val_x, val_y = np.load(Path(model_dir)/'train_x_windows.npy'), np.load(Path(model_dir)/'train_y_windows.npy'), \
                np.load(Path(model_dir)/'val_x_windows.npy'), np.load(Path(model_dir)/'val_y_windows.npy')

        make_balanced = False

        if make_balanced:
            train_x, train_y, val_x, val_y = balance_data(train_x, train_y, val_x, val_y, window_size)
            print("Data is balanced")
            print("Nr of examples per label in training data: ", len(train_x)/3)
            print("Nr of examples per label in validation data: ", len(val_x)/3)
            
        print("Nr of taining samples: ", len(train_x))
        print("Nr of validation samples: ", len(val_x))

    else:
        print('Retrieving the labels')
        train_y, val_y = load_all_labels(df_train, shift=0), load_all_labels(df_val, shift=0)
        
        print('Calculating training vector derivatives')
        train_x, val_x = derivatives_from_csv(df_train), derivatives_from_csv(df_val)

        print("Removing files that do not contain nods from training and validation data") # these may contain nods labeled as background
        indexes_with_nods_train = [index for index, array in enumerate(train_y) if 2 in array]
        indexes_with_nods_val = [index for index, array in enumerate(val_y) if 2 in array]
        print("Nr of files in training data that contain nods: ", len(indexes_with_nods_train), "/", len(train_y), ", namely the indexes: ", indexes_with_nods_train)
        print("Nr of files in validation data that contain nods: ", len(indexes_with_nods_val), "/", len(val_y), ", namely the indexes: ", indexes_with_nods_val)
        train_x, train_y = [train_x[index] for index in indexes_with_nods_train], [train_y[index] for index in indexes_with_nods_train]
        val_x, val_y = [val_x[index] for index in indexes_with_nods_val], [val_y[index] for index in indexes_with_nods_val]

        print('Separating sequences into windows with frame-by-frame labels')
        train_background_vectors, train_shake_vectors, train_nod_vectors, train_shake_label_seqs, train_nod_label_seqs, train_background_label_seqs = separate_seqs_all_with_return_seq(train_x, train_y, window_size)
        val_background_vectors, val_shake_vectors, val_nod_vectors, val_shake_label_seqs, val_nod_label_seqs, val_background_label_seqs = separate_seqs_all_with_return_seq(val_x, val_y, window_size)

        print('Yaw, pitch, roll graphs')
        plot_pyr_graphs(train_shake_vectors, number=1, add_diff=False, diff_pix=False, diff_ang=False, movement_type='Shake')
        plot_pyr_graphs(train_nod_vectors, number=1, add_diff=False, diff_pix=False, diff_ang=False, movement_type='Nod')

        # print('Augmenting data')
        # train_nod_vectors, _ = augment_flip_data(train_nod_vectors, [2]*len(train_nod_vectors))
        # val_nod_vectors, _ = augment_flip_data(val_nod_vectors, [2]*len(val_nod_vectors))

        print('Check number of data points: ')
        min_train = min(len(train_background_vectors), len(train_shake_vectors), len(train_nod_vectors))
        min_val = min(len(val_background_vectors), len(val_shake_vectors), len(val_nod_vectors))
        print("Min train: ", min_train, ", min val: ", min_val)

        print('Shuffle all training and validation sets separately')
        # min_train, min_val = -1, -1       # to disable min_val and min_train
        train_background_vectors, train_background_label_seqs = shuffle_data(train_background_vectors, train_background_label_seqs, nr_samples=min_train)
        train_shake_vectors, train_shake_label_seqs = shuffle_data(train_shake_vectors, train_shake_label_seqs, nr_samples=min_train)
        train_nod_vectors, train_nod_label_seqs = shuffle_data(train_nod_vectors, train_nod_label_seqs, nr_samples=min_train)
        val_background_vectors, val_background_label_seqs = shuffle_data(val_background_vectors, val_background_label_seqs, nr_samples=min_val)
        val_shake_vectors, val_shake_label_seqs = shuffle_data(val_shake_vectors, val_shake_label_seqs, nr_samples=min_val)
        val_nod_vectors, val_nod_label_seqs = shuffle_data(val_nod_vectors, val_nod_label_seqs, nr_samples=min_val)

        print('Concatenate')
        train_x = np.array(train_background_vectors + train_shake_vectors + train_nod_vectors)
        train_y = np.array(train_background_label_seqs + train_shake_label_seqs + train_nod_label_seqs)
        val_x = np.array(val_background_vectors + val_shake_vectors + val_nod_vectors)
        val_y = np.array(val_background_label_seqs + val_shake_label_seqs + val_nod_label_seqs)

        print('Shuffle training and validation data')
        (train_x, train_y), (val_x, val_y) = shuffle_data(train_x, train_y), shuffle_data(val_x, val_y)
        print("Nr of taining samples: ", len(train_x))
        print("Nr of validation samples: ", len(val_x))
        
        print('Convert data to numpy arrays')
        train_x, train_y, val_x, val_y = np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)

        print('Scaling data')
        from sklearn.preprocessing import StandardScaler
        # create a scaler, one for each feature (yaw, pitch, roll, shoulder)
        scalers = {}
        for i in range(train_x.shape[2]):
            scalers[i] = StandardScaler()
            train_x[:, :, i] = scalers[i].fit_transform(train_x[:, :, i].reshape(-1, 1)).reshape(train_x[:, :, i].shape)
        for i in range(val_x.shape[2]):
            val_x[:, :, i] = scalers[i].transform(val_x[:, :, i].reshape(-1, 1)).reshape(val_x[:, :, i].shape)

        print('Saving scalers')
        joblib.dump(scalers, Path(model_dir)/'scalers.pkl')

        np.save(Path(model_dir) / Path(f'train_x_windows.npy'), train_x)
        np.save(Path(model_dir) / Path(f'train_y_windows.npy'), train_y)
        np.save(Path(model_dir) / Path(f'val_x_windows.npy'), val_x)
        np.save(Path(model_dir) / Path(f'val_y_windows.npy'), val_y)


    plot_pyr_graphs(train_x, number=1, add_diff=False, diff_pix=False, diff_ang=False, movement_type='Unk')

    print('Make dictionaries')
    simple_sign_dict = {'background': 0, 'shake': 1, 'nod': 2}
    nr_features = len(train_x[0][0])

    nr_classes = len(simple_sign_dict)
    print("Nr of classes = ", nr_classes)

    callbacks1 = ModelCheckpoint('data/multiple_model.hdf5', save_best_only=True)
    callbacks2 = EarlyStopping(
        monitor='val_loss', 
        patience = 10, 
        verbose = 1,
        restore_best_weights=True)
    callbacks3 = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=2,
        verbose=1
    )

    callbacks = [callbacks1, callbacks2, callbacks3]

    print('Creating model')    
    model = Sequential()
    model.add(Bidirectional(LSTM(units=64, return_sequences=True, kernel_regularizer=l2(0.02)), input_shape=(window_size, nr_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.02))))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(units=nr_classes, activation='softmax')))

    print(model.summary())

    # optimizer = Adam(learning_rate=0.005)
    # lr_schedule = ExponentialDecay(
    # initial_learning_rate=0.05,
    # decay_steps=10000,
    # decay_rate=0.9)
    # optimizer = SGD(learning_rate=lr_schedule)
    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy")

    # Add class weights for unbalanced training
    # flat_labels = [label for sublist in train_y for label in sublist]
    # class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(flat_labels), y=flat_labels)
    # class_weights = dict(enumerate(class_weights))
    # print("Class weights: ", class_weights)
    
    train_x = train_x.tolist()
    val_x = val_x.tolist()
    train_y = train_y.tolist()  
    val_y = val_y.tolist()
    
    history = model.fit(train_x, train_y, validation_data=[val_x, val_y], epochs=10, callbacks=callbacks, batch_size=16)

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

    # predict on validation set
    predictions = model.predict(val_x)
    predictions = np.argmax(predictions, axis=-1)
    predictions = [sublist[math.floor(window_size/2)] for sublist in predictions]
    output = [sublist[math.floor(window_size/2)] for sublist in val_y]
    print(output[0:10])
    print(predictions[0:10])
    print("Accuracy: ", accuracy_score(output, predictions))
    conf_matrix = confusion_matrix(output, predictions)
    plot_confusion_matrix(conf_matrix, title=None, labels = simple_sign_dict)

    print("Saving model")
    model.save(model_dir/MODEL_LSTM_CLASS_FILENAME)

    # # predict on validation set
    # from models.lstm.test_lstm_class_windows import predict_lstm_class_windows
    # print("predict on saved validation data")
    # output, predictions = predict_lstm_class_windows(df_val, model_dir, window_size=36, load_data=True)
    # print(output[0:10])
    # print(predictions[0:10])
    # print("Accuracy: ", accuracy_score(output, predictions))

    # output, predictions = predict_lstm_class_windows(df_val, model_dir, window_size=36, load_data=False, random_windows=True)
    # print("predict on all validation data")
    # print(output[0:10])
    # print(predictions[0:10])
    # print("Accuracy: ", accuracy_score(output, predictions))
    # conf_matrix = metrics.confusion_matrix(output, predictions)
    # plot_confusion_matrix(conf_matrix, title=None, labels = simple_sign_dict)

    print('Training program complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    fit_lstms_class_windows(args.frames_csv, args.model_dir, args.overwrite)

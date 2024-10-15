import argparse
import pickle
from pathlib import Path

import numpy as np
import torch 
import random
import joblib

from models.processing.preparation import separate_seqs, remove_nan_values_from_tensors, separate_seqs_all, separate_seqs_all_middle_frame_all, plot_pyr_graphs, shuffle_data
from models.processing.facial_movement import derivatives_from_csv
from utils.frames_csv import load_df, load_all_labels
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from validation.validate_model import plot_confusion_matrix
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import math

MINIMUM_SEQUENCE_LENGTH = 2
MODEL_LSTM_CLASS_FILENAME = 'lstm_class_bi_windows_middle_nod.h5'
MASK_VALUE = [-10]*4
PAD_VALUE = [0]*4

import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers.legacy import SGD
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Masking, Bidirectional, Dropout, Input, Attention, concatenate, Lambda
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.regularizers import l2

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint


def fit_lstms_class_windows_middle_nod(df_train, df_val, model_dir, overwrite=True, debug=False, epochs=100, window_size=36, load_values=False):
    if (Path(model_dir)/MODEL_LSTM_CLASS_FILENAME).exists() and not overwrite:
        if not debug:
            raise FileExistsError('The models already exist in this directory, you can explicitly allow overwriting')

    if load_values and (Path(model_dir)/'train_x_windows_middle_nod.npy').exists() and (Path(model_dir)/'val_x_windows_middle_nod.npy').exists()\
                   and (Path(model_dir)/'train_y_windows_middle_nod.npy').exists() and (Path(model_dir)/'val_y_windows_middle_nod.npy').exists():
        print('Loading train and val data from pickle')
        train_x, train_y, val_x, val_y = np.load(Path(model_dir)/'train_x_windows_middle_nod.npy'), np.load(Path(model_dir)/'train_y_windows_middle_nod.npy'), \
                np.load(Path(model_dir)/'val_x_windows_middle_nod.npy'), np.load(Path(model_dir)/'val_y_windows_middle_nod.npy')
    else:
        print('Retrieving the training labels')
        train_y = load_all_labels(df_train, shift=0)
        
        print('Calculating training vector derivatives')
        train_x = derivatives_from_csv(df_train)

        print("Removing files that do not contain nods")
        indexes_with_nods = [index for index, array in enumerate(train_y) if 2 in array]
        print("Nr of files in training data that contain nods: ", len(indexes_with_nods), "/", len(train_y), ", namely the indexes: ", indexes_with_nods)
        train_y = [train_y[index] for index in indexes_with_nods]
        train_x = [train_x[index] for index in indexes_with_nods]

        print('Separating training sequences')
        train_background_vectors, train_shake_vectors, train_nod_vectors, _, _, _ = separate_seqs_all_middle_frame_all(train_x, train_y, window_size)
        train_background_vectors += train_shake_vectors
        train_background_labels = [0]*len(train_background_vectors)
        train_nod_labels = [1]*len(train_nod_vectors)

        print('Plotting some graphs')
        plot_pyr_graphs(train_nod_vectors, number=1, add_diff = False, diff_pix=False, diff_ang=False, movement_type='Nod', ang_norm=False)

        print('Retrieving the validation labels')
        val_y = load_all_labels(df_val, shift=0)
        print('Calculating validation vector derivatives')
        val_x = derivatives_from_csv(df_val)

        print("Removing files that do not contain nods")
        indexes_with_nods = [index for index, array in enumerate(val_y) if 2 in array]
        print("Nr of files in validation data that contain nods: ", len(indexes_with_nods), "/", len(val_y), ", namely the indexes: ", indexes_with_nods)
        val_y = [val_y[index] for index in indexes_with_nods]
        val_x = [val_x[index] for index in indexes_with_nods]

        print('Separating validation sequences')
        val_background_vectors, val_shake_vectors, val_nod_vectors, _, _, _ = separate_seqs_all_middle_frame_all(val_x, val_y, window_size)
        val_background_vectors += val_shake_vectors
        val_background_labels = [0]*len(val_background_vectors)
        val_nod_labels = [1]*len(val_nod_vectors)

        # print('Augmenting data')
        # train_nod_vectors, _ = augment_flip_data(train_nod_vectors, [2]*len(train_nod_vectors))
        # val_nod_vectors, _ = augment_flip_data(val_nod_vectors, [2]*len(val_nod_vectors))

        print('Check number of data points: ')
        min_train, min_val = len(train_nod_vectors), len(val_nod_vectors)
        print("Min train: ", min_train, ", min val: ", min_val)

        print('Shuffle all training and validation sets separately')
        #disable min_val and min_train
        # min_train, min_val = -1, -1
        train_background_vectors, train_background_labels = shuffle_data(train_background_vectors, train_background_labels, nr_samples=min_train)
        train_nod_vectors, train_nod_labels = shuffle_data(train_nod_vectors, train_nod_labels, nr_samples=min_train)
        val_background_vectors, val_background_labels = shuffle_data(val_background_vectors, val_background_labels, nr_samples=min_val)
        val_nod_vectors, val_nod_labels = shuffle_data(val_nod_vectors, val_nod_labels, nr_samples=min_val)

        print('Concatenate')
        train_x = np.array(train_background_vectors + train_nod_vectors)
        train_y = np.array(train_background_labels + train_nod_labels)
        val_x = np.array(val_background_vectors + val_nod_vectors)
        val_y = np.array(val_background_labels + val_nod_labels)

        print('Shuffle training and validation data')
        train_x, train_y = shuffle_data(train_x, train_y)
        val_x, val_y = shuffle_data(val_x, val_y)
        print("Nr of validation samples: ", len(val_x))
        
        print('Convert data to numpy arrays')
        train_x, train_y, val_x, val_y = np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)
        print(train_y.shape)

        print('Scaling data')
        from sklearn.preprocessing import StandardScaler
        scalers = {}
        for i in range(train_x.shape[2]):
            scalers[i] = StandardScaler()
            train_x[:, :, i] = scalers[i].fit_transform(train_x[:, :, i].reshape(-1, 1)).reshape(train_x[:, :, i].shape)
        print(train_x.shape)

        for i in range(val_x.shape[2]):
            val_x[:, :, i] = scalers[i].transform(val_x[:, :, i].reshape(-1, 1)).reshape(val_x[:, :, i].shape)
        print(val_x.shape)

        print('Saving scalers')
        joblib.dump(scalers, Path(model_dir)/'scalers_middle_nod.pkl')

        np.save(Path(model_dir) / Path(f'train_x_windows_middle_nod.npy'), train_x)
        np.save(Path(model_dir) / Path(f'train_y_windows_middle_nod.npy'), train_y)
        np.save(Path(model_dir) / Path(f'val_x_windows_middle_nod.npy'), val_x)
        np.save(Path(model_dir) / Path(f'val_y_windows_middle_nod.npy'), val_y)

    ## UNTIL HERE in else

    plot_pyr_graphs(train_x, number=1, add_diff = False, diff_pix=False, diff_ang=False, movement_type='Unk', ang_norm=False)

    print('Make dictionaries')
    simple_sign_dict = {'background': 0, 'nod': 1}
    turned_simple_sign_dict = {}
    for ryi, ry in enumerate(list(set(simple_sign_dict))):
        turned_simple_sign_dict[ryi] = ry
    # print(turned_simple_sign_dict)
    
    nr_features = len(train_x[0][0])

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

    callbacks1 = ModelCheckpoint('data/class_windows_middle_nod.hdf5', save_best_only=True)
    callbacks2 = EarlyStopping(
        monitor='val_loss', 
        patience = 10, 
        verbose = 1,
        restore_best_weights=True)
    callbacks3 = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    callbacks = [callbacks1, callbacks2, callbacks3]

    # Define the model
    # model = Sequential()
    # model.add(Masking(mask_value=MASK_VALUE, input_shape=(window_size, nr_features)))  
    # model.add(Bidirectional(LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.01))))  
    # model.add(LSTM(units=16, return_sequences=False, kernel_regularizer=l2(0.01)))  
    # model.add(Dropout(0.5))
    # model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)))  
    # model.add(Dropout(0.5))  
    # model.add(Dense(units=nr_classes, activation='softmax'))

    from keras.models import Sequential
    from keras.layers import Masking, Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, GRU, Dropout, Dense
    from keras.regularizers import l2

    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(window_size, nr_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(GRU(units=32, return_sequences=False, kernel_regularizer=l2(0.001), recurrent_dropout=0.2)))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(units=nr_classes, activation='softmax'))

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy")
    
    print(model.summary())
    
    train_x = train_x.tolist()
    val_x = val_x.tolist()
    train_y = train_y.tolist()
    val_y = val_y.tolist()

    history = model.fit(train_x, train_y, validation_data=[val_x, val_y], epochs=15, callbacks=callbacks, batch_size=32)

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
    predictions = np.argmax(predictions, axis=1)
    print("Accuracy: ", accuracy_score(val_y, predictions))
    confusion_matrix = metrics.confusion_matrix(val_y, predictions)
    plot_confusion_matrix(confusion_matrix, title=None, labels = simple_sign_dict)

    print("Saving model")
    model.save(model_dir/MODEL_LSTM_CLASS_FILENAME)

    print('Split complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    fit_lstms_class_windows_middle_nod(args.frames_csv, args.model_dir, args.overwrite)

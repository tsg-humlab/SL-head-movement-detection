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
MODEL_LSTM_CLASS_FILENAME = 'lstm_class_bi_windows_middle_all.h5'
MASK_VALUE = [-10]*4
PAD_VALUE = [0]*4

import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers.legacy import SGD
from keras.optimizers.schedules import ExponentialDecay
from keras.models import Sequential, Model
from keras.layers import Embedding, LSTM, Dense, TimeDistributed, Masking, Bidirectional, Dropout, Input, Attention, concatenate, Lambda, BatchNormalization
from keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from keras.regularizers import l2

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint

from models.processing.preparation import add_diffs_list

def plot_pyr_graphs(vectors, number=3, add_diff=False, diff_pix=False, diff_ang=False, movement_type='Movement', ang_norm=False):
    from matplotlib import pyplot as plt
    import random 
    example_vectors = random.sample(range(0,len(vectors)), number)
    for vector_i, vector in enumerate(vectors):
        if vector_i in example_vectors:
            p, y, r, s = [], [], [], []
            if add_diff:
                vector = add_diffs_list(vector)
            else:
                vector = vector
            for xi, xx in enumerate(vector):
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
            plt.figsize=(30,6)
            plt.plot(p, label='pitch')
            plt.plot(y, label='yaw')
            plt.plot(r, label='roll')
            plt.plot(s, label='shoulder')
            plt.ylabel('angles')
            plt.xlabel('frame')
            plt.title(movement_type + " angles")
            plt.legend()
            plt.show()


def fit_lstms_class_windows_middle_all(df_train, df_val, model_dir, overwrite=True, debug=False, epochs=100, window_size=36, load_values=False):
    if (Path(model_dir)/MODEL_LSTM_CLASS_FILENAME).exists() and not overwrite:
        if not debug:
            raise FileExistsError('The models already exist in this directory, you can explicitly allow overwriting')

    if load_values and (Path(model_dir)/'train_x_windows_middle_all.npy').exists() and (Path(model_dir)/'val_x_windows_middle_all.npy').exists()\
                   and (Path(model_dir)/'train_y_windows_middle_all.npy').exists() and (Path(model_dir)/'val_y_windows_middle_all.npy').exists():
        print('Loading train and val data from pickle')
        train_x, train_y, val_x, val_y = np.load(Path(model_dir)/'train_x_windows_middle_all.npy'), np.load(Path(model_dir)/'train_y_windows_middle_all.npy'), \
                np.load(Path(model_dir)/'val_x_windows_middle_all.npy'), np.load(Path(model_dir)/'val_y_windows_middle_all.npy')
    
    else:
        print('Retrieving the training labels')
        train_y = load_all_labels(df_train, shift=0)
        
        print('Calculating training vector derivatives')
        train_x = derivatives_from_csv(df_train)

        print("Removing files that do not contain nods")
        indexes_with_nods = [index for index, array in enumerate(train_y) if 2 in array]
        print("Nr of files in training data that contain shakes: ", len(indexes_with_nods), "/", len(train_y), ", namely the indexes: ", indexes_with_nods)
        train_y = [train_y[index] for index in indexes_with_nods]
        train_x = [train_x[index] for index in indexes_with_nods]

        print('Separating training sequences')
        train_background_vectors, train_shake_vectors, train_nod_vectors, train_shake_labels, train_background_labels, train_nod_labels = separate_seqs_all_middle_frame_all(train_x, train_y, window_size)

        print('Retrieving the validation labels')
        val_y = load_all_labels(df_val, shift=0)
        print('Calculating validation vector derivatives')
        val_x = derivatives_from_csv(df_val)

        print("Removing files that do not contain nods")
        indexes_with_nods = [index for index, array in enumerate(val_y) if 2 in array]
        print("Nr of files in validation data that contain shakes: ", len(indexes_with_nods), "/", len(val_y), ", namely the indexes: ", indexes_with_nods)
        val_y = [val_y[index] for index in indexes_with_nods]
        val_x = [val_x[index] for index in indexes_with_nods]

        print('Separating validation sequences')
        val_background_vectors, val_shake_vectors, val_nod_vectors, val_shake_labels, val_background_labels, val_nod_labels = separate_seqs_all_middle_frame_all(val_x, val_y, window_size)

        # print('Augmenting data')
        # train_nod_vectors, _ = augment_flip_data(train_nod_vectors, [2]*len(train_nod_vectors))
        # val_nod_vectors, _ = augment_flip_data(val_nod_vectors, [2]*len(val_nod_vectors))

        print('Check number of data points: ')
        min_train, min_val = min(len(train_nod_vectors), len(train_shake_vectors)), min(len(val_nod_vectors), len(val_shake_vectors))
        print("Min train: ", min_train, ", min val: ", min_val)

        print('Shuffle all training and validation sets separately')
        # min_train, min_val = -1, -1   #to disable min_val and min_train
        train_background_vectors, train_background_labels = shuffle_data(train_background_vectors, train_background_labels, nr_samples=min_train)
        train_shake_vectors, train_shake_labels = shuffle_data(train_shake_vectors, train_shake_labels, nr_samples=min_train)
        train_nod_vectors, train_nod_labels = shuffle_data(train_nod_vectors, train_nod_labels, nr_samples=min_train)
        val_background_vectors, val_background_labels = shuffle_data(val_background_vectors, val_background_labels, nr_samples=min_val)
        val_shake_vectors, val_shake_labels = shuffle_data(val_shake_vectors, val_shake_labels, nr_samples=min_val)
        val_nod_vectors, val_nod_labels = shuffle_data(val_nod_vectors, val_nod_labels, nr_samples=min_val)

        print('Concatenate')
        train_x = np.array(train_background_vectors + train_shake_vectors + train_nod_vectors)
        train_y = np.array(train_background_labels + train_shake_labels + train_nod_labels)
        val_x = np.array(val_background_vectors + val_shake_vectors + val_nod_vectors)
        val_y = np.array(val_background_labels + val_shake_labels + val_nod_labels)

        print('Shuffle training and validation data')
        train_x, train_y = shuffle_data(train_x, train_y)
        val_x, val_y = shuffle_data(val_x, val_y)
        print("Nr of validation samples: ", len(val_x))
        
        print('Convert data to numpy arrays')
        train_x, train_y, val_x, val_y = np.array(train_x), np.array(train_y), np.array(val_x), np.array(val_y)
        print(train_y.shape)

        print('Plot some signals before scaling')
        train_shake_indexes = [index for index, array in enumerate(train_y) if array == 1]
        train_nod_indexes = [index for index, array in enumerate(train_y) if array == 2]
        train_shake_vectors = [train_x[index] for index in train_shake_indexes]
        train_nod_vectors = [train_x[index] for index in train_nod_indexes]
        plot_pyr_graphs(train_shake_vectors[0:1], add_diff = False, diff_pix=False, diff_ang=False, movement_type='Shake')
        plot_pyr_graphs(train_nod_vectors[0:1], add_diff = False, diff_pix=False, diff_ang=False, movement_type='Nod')

        print('Scaling data')
        from sklearn.preprocessing import StandardScaler
        scalers = {}
        for i in range(train_x.shape[2]):
            scalers[i] = StandardScaler()
            train_x[:, :, i] = scalers[i].fit_transform(train_x[:, :, i].reshape(-1, 1)).reshape(train_x[:, :, i].shape)
        print(train_x.shape)

        # print('Plot the same signals after scaling')
        # plot_pyr_graphs(train_shake_vectors[0:1], add_diff = False, diff_pix=False, diff_ang=False, movement_type='Shake')
        # plot_pyr_graphs(train_nod_vectors[0:1], add_diff = False, diff_pix=False, diff_ang=False, movement_type='Nod')

        for i in range(val_x.shape[2]):
            val_x[:, :, i] = scalers[i].transform(val_x[:, :, i].reshape(-1, 1)).reshape(val_x[:, :, i].shape)
        print(val_x.shape)

        print('Saving scalers')
        joblib.dump(scalers, Path(model_dir)/'scalers_middle_all.pkl')

        np.save(Path(model_dir) / Path(f'train_x_windows_middle_all.npy'), train_x)
        np.save(Path(model_dir) / Path(f'train_y_windows_middle_all.npy'), train_y)
        np.save(Path(model_dir) / Path(f'val_x_windows_middle_all.npy'), val_x)
        np.save(Path(model_dir) / Path(f'val_y_windows_middle_all.npy'), val_y)
    
    print('Make dictionaries')
    simple_sign_dict = {'background': 0, 'shake': 1, 'nod': 2}
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

    callbacks1 = ModelCheckpoint('data/class_windows_middle_all.hdf5', save_best_only=True)
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
    # model.add(Masking(mask_value=MASK_VALUE, input_shape=(window_size, nr_features)))  # Masking layer to handle variable sequence lengths
    # # Add LSTM layers to capture temporal dependencies
    # model.add(Bidirectional(LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.04), recurrent_dropout=0.2)))  # Specify input_shape in the first layer
    # model.add(LSTM(units=16, return_sequences=False, kernel_regularizer=l2(0.04)))  # You can adjust the number of units and regularization as needed
    # model.add(Dropout(0.5))
    # # Add Dense layers for classification
    # # model.add(Dense(units=64, activation='relu', kernel_regularizer=l2(0.04)))  # Added Dense layer with L2 regularization
    # # model.add(Dropout(0.5))  # Additional Dropout layer for more regularization
    # model.add(Dense(units=nr_classes, activation='softmax'))


    from keras.models import Sequential
    from keras.layers import Masking, Conv1D, MaxPooling1D, Bidirectional, GRU, Dropout, Dense, GlobalMaxPooling1D
    from keras.regularizers import l2
    from keras.layers import BatchNormalization

    model = Sequential()
    model.add(Masking(mask_value=MASK_VALUE, input_shape=(window_size, nr_features)))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', 
                    kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(GRU(units=16, return_sequences=False, 
                                kernel_regularizer=l2(0.001), 
                                recurrent_dropout=0.2)))

    model.add(Dropout(0.5))
    model.add(Dense(units=32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(units=nr_classes, activation='softmax'))

    print(model.summary())

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics="accuracy")
    
    train_x = train_x.tolist()
    val_x = val_x.tolist()
    train_y = train_y.tolist()
    val_y = val_y.tolist()

    history = model.fit(train_x, train_y, validation_data=[val_x, val_y], epochs=30, callbacks=callbacks, batch_size=32)

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
    predictions = np.argmax(predictions, axis=1).tolist()
    print("Accuracy: ", accuracy_score(val_y, predictions))
    confusion_matrix = metrics.confusion_matrix(val_y, predictions, labels=[0,1,2])
    plot_confusion_matrix(confusion_matrix, title=None, labels = simple_sign_dict)

    print("Saving model")
    model.save(model_dir/MODEL_LSTM_CLASS_FILENAME)

    # # predict on validation set
    from models.lstm.test_lstm_class_windows_middle_all import predict_lstm_class_windows_middle_all
    # print("predict on saved validation data")
    # output, predictions = predict_lstm_class_windows_middle(df_val, model_dir, window_size=36, load_data=True)
    # print(output[0:10])
    # print(predictions[0:10])
    # print("Accuracy: ", accuracy_score(output, predictions))

    output, predictions = predict_lstm_class_windows_middle_all(df_val, model_dir, window_size=36, load_data=False, random_windows=True)
    print("predict on all validation data")
    print("Accuracy: ", accuracy_score(output, predictions))
    confusion_matrix = metrics.confusion_matrix(output, predictions)
    plot_confusion_matrix(confusion_matrix, title=None, labels = simple_sign_dict)

    print('Split complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    fit_lstms_class_windows_middle_all(args.frames_csv, args.model_dir, args.overwrite)

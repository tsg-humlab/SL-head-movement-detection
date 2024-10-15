import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from pomegranate.distributions import Normal
from pomegranate.hmm import DenseHMM

import matplotlib.pyplot as plt

from models.processing.facial_movement import derivatives_from_csv
from utils.frames_csv import load_df, load_all_labels

MINIMUM_SEQUENCE_LENGTH = 2
MODEL_LSTM_FILENAME = 'lstm.h5'
WEIGHTS_LSTM_FILENAME = 'lstm_weights.h5'
MAXLEN = 20000

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, TimeDistributed, Masking, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.losses import CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC


def fit_lstms(df_train, df_val, model_dir, pad_length=20000, overwrite=True, debug=False, epochs=100):
    if ((Path(model_dir)/MODEL_LSTM_FILENAME).exists()) and not overwrite:
        if not debug:
            raise FileExistsError('The models already exist in this directory, you can explicitly allow overwriting')

    print('Calculating vector derivatives')
    train_y = load_all_labels(df_train, shift=1)
    train_x = derivatives_from_csv(df_train)
    val_x = derivatives_from_csv(df_val)
    val_y = load_all_labels(df_val, shift=1)

    print("Removing files that do not contain nods")
    indexes_with_nods = [index for index, array in enumerate(train_y) if 2 in array]
    print("Nr of files in training data that contain nods: ", len(indexes_with_nods), "/", len(train_y), ", namely the indexes: ", indexes_with_nods)
    train_y = [train_y[index] for index in indexes_with_nods]
    train_x = [train_x[index] for index in indexes_with_nods]
    indexes_with_nods = [index for index, array in enumerate(val_y) if 2 in array]
    print("Nr of files in validation data that contain nods: ", len(indexes_with_nods), "/", len(val_y), ", namely the indexes: ", indexes_with_nods)
    val_y = [val_y[index] for index in indexes_with_nods]
    val_x = [val_x[index] for index in indexes_with_nods]

    print("calculate weights for each label")
    class_weights = []
    conc_train_y = np.concatenate(train_y)
    class_weights.append(1/np.sum(conc_train_y == 0))
    class_weights.append(1/np.sum(conc_train_y == 1))
    class_weights.append(1/np.sum(conc_train_y == 2))
    norm_class_weights = [float(i)/sum(class_weights) for i in class_weights]
    class_weight_dict = {0: norm_class_weights[0], 1: norm_class_weights[1], 2: norm_class_weights[2]}
    weight_arr = []
    longest_sequence = 0
    for y_t in train_y:
        weight_arr.append([class_weight_dict[weight] for weight in y_t])
        if len(y_t) > longest_sequence:
            longest_sequence = len(y_t)
    print("Longest sequence: ", longest_sequence)

    print("Padding all sequences to the same length")
    special_value = -10 # Value to mask out
    train_x = tf.keras.preprocessing.sequence.pad_sequences(train_x, padding='post', maxlen=MAXLEN, dtype='float32', value=[special_value]*len(train_x[0][0]))
    val_x = tf.keras.preprocessing.sequence.pad_sequences(val_x, padding='post', maxlen=MAXLEN, dtype='float32', value=[special_value]*len(train_x[0][0]))
    train_y = tf.keras.preprocessing.sequence.pad_sequences(train_y, padding='post', maxlen=MAXLEN, value = 0)
    val_y = tf.keras.preprocessing.sequence.pad_sequences(val_y, padding='post', maxlen=MAXLEN, value = 0)
    sample_weight = tf.keras.preprocessing.sequence.pad_sequences(weight_arr, maxlen=MAXLEN, dtype='float32', padding='post', value = 0)

    print("Split into sequences of certain length")
    sequence_length = 2000
    new_train_x, new_train_y, new_sample_weights, new_val_x, new_val_y = [], [], [], [], []
    for i in range(len(train_x)):
        new_train_x += [train_x[i][j:j+sequence_length] for j in range(0, len(train_x[i]), sequence_length)]
        new_train_y += [train_y[i][j:j+sequence_length] for j in range(0, len(train_y[i]), sequence_length)]
        new_sample_weights += [sample_weight[i][j:j+sequence_length] for j in range(0, len(sample_weight[i]), sequence_length)]
    print("Length of new training and sample_weights: ", len(new_train_x), len(new_train_y), len(new_sample_weights))
    print(f'"This should be of length {len(train_x)} * {MAXLEN} / {sequence_length}: {len(train_x)*(MAXLEN/sequence_length)}"')
    assert len(new_train_x) == len(new_train_y) == len(new_sample_weights) == int(len(train_x)*MAXLEN/sequence_length)
    for i in range(len(val_x)):
        new_val_x += [val_x[i][j:j+sequence_length] for j in range(0, len(val_x[i]), sequence_length)]
        new_val_y += [val_y[i][j:j+sequence_length] for j in range(0, len(val_y[i]), sequence_length)]
    train_x, train_y, sample_weight, val_x, val_y = np.array(new_train_x), np.array(new_train_y), np.array(new_sample_weights), np.array(new_val_x), np.array(new_val_y)

    print("Remove sequences with all labels 0")
    indexes_to_remove = [index for index, array in enumerate(train_y) if np.all(array == 0)]
    train_x = np.delete(train_x, indexes_to_remove, axis=0)
    train_y = np.delete(train_y, indexes_to_remove, axis=0)
    sample_weight = np.delete(sample_weight, indexes_to_remove, axis=0)
    indexes_to_remove = [index for index, array in enumerate(val_y) if np.all(array == 0)]
    val_x = np.delete(val_x, indexes_to_remove, axis=0)
    val_y = np.delete(val_y, indexes_to_remove, axis=0)

    print("One hot encoding")
    one_hot_train_y = tf.keras.utils.to_categorical(train_y, num_classes=3)
    one_hot_val_y = tf.keras.utils.to_categorical(val_y, num_classes=3)

    print("Building model")
    num_labels = 3  # Nr of unique labels
    embedding_dim = 4 # Nr of values per timestep = nr of features
    
    model = Sequential()
    model.add(Masking(mask_value=[special_value]*embedding_dim, input_shape=(sequence_length, embedding_dim)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3)) 
    model.add(Bidirectional(LSTM(32, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.3)) 
    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3)) 
    model.add(Dense(num_labels, activation='softmax'))
    # model.add(TimeDistributed(Dense(16)))
    # model.add(TimeDistributed(Dense(num_labels, activation='softmax')))
    model.summary()

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=1e-6, verbose=1)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    print("Compiling model")
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(
        loss=CategoricalCrossentropy(),
        # optimizer="sgd",
        optimizer=optimizer,
        metrics=["accuracy", AUC(curve='ROC'), AUC(curve='PR')],
        sample_weight_mode="temporal",
    )

    print("Training model")
    history = model.fit(
        train_x, one_hot_train_y, validation_data=(val_x, one_hot_val_y), epochs=50, batch_size=1, sample_weight=sample_weight, callbacks=[reduce_lr, early_stop]
    )
    
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

    print("Saving model")
    model.save(model_dir/MODEL_LSTM_FILENAME)
    # model.save_weights(model_dir/WEIGHTS_LSTM_FILENAME)

    print('Training program complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('model_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    fit_lstms(args.frames_csv, args.model_dir, args.overwrite)

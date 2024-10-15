import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from models.processing.facial_movement import derivatives_from_csv
from models.lstm.train_lstm import MODEL_LSTM_FILENAME, WEIGHTS_LSTM_FILENAME, MAXLEN
from models.simple.detector import verify_window_size
from utils.frames_csv import load_df, load_all_labels

import tensorflow as tf
from tensorflow.keras.models import Sequential

def predict_lstm(df_test, model_dir):

    # load test data
    df_test = load_df(df_test)
    test_x = derivatives_from_csv(df_test)
    test_y = load_all_labels(df_test, shift=1)

    # load model
    model = tf.keras.models.load_model(model_dir/MODEL_LSTM_FILENAME)
    
    special_value = -10 # Value to mask out
    test_x = tf.keras.preprocessing.sequence.pad_sequences(test_x, padding='post', maxlen=MAXLEN, dtype='float32', value = [special_value, special_value, special_value])

    # predict on test data
    preds = model.predict(test_x)
    predictions = []
    for pred in preds:
        predictions.append(np.argmax(pred, axis=1))

    print("Predictions: ", predictions[0][100:200])
    print("Test_y: ", test_y[0][100:200])

    return test_y, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('models_dir', type=Path)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    predict_lstm(args.frames_csv, args.models_dir)

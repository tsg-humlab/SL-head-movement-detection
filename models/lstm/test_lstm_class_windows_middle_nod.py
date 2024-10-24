import argparse
import pickle
from pathlib import Path
import math
import joblib

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from models.processing.facial_movement import derivatives_from_csv
from models.lstm.train_lstm_class_windows_middle_nod import MODEL_LSTM_CLASS_FILENAME, PAD_VALUE, MASK_VALUE, plot_pyr_graphs
from models.simple.detector import verify_window_size
from utils.frames_csv import load_df, load_all_labels
from keras.models import Model

import tensorflow as tf

def predict_lstm_class_windows_middle_nod(df_test, models_dir, window_size=36):

    window_size = verify_window_size(window_size)
    
    # load model
    model = tf.keras.models.load_model(models_dir/MODEL_LSTM_CLASS_FILENAME)
    
    # initialize predictions and labels
    predictions, labels = [], []

    print("Creating data")
    df_test = load_df(df_test)
    vectors = derivatives_from_csv(df_test).copy()
    labels = load_all_labels(df_test, shift=0, window=window_size)

    print("Removing files that do not contain nods")
    indexes_with_nods = [index for index, array in enumerate(labels) if 2 in array]
    print("Nr of files in validation data that contain nods: ", len(indexes_with_nods), "/", len(labels), ", namely the indexes: ", indexes_with_nods)
    labels = [labels[index] for index in indexes_with_nods]
    vectors = [vectors[index] for index in indexes_with_nods]

    import random
    to_show_pyr = random.sample(range(len(vectors)), 3)

    # for each video
    for vector_i, vector in enumerate(vectors):
        
        # make windows
        windows = []

        # get nr of predictions
        pred_len = len(vector) - window_size + 1

        # Make windows of (masked/padded) length
        for i in range(pred_len):
            windows.append(vector[i:i + window_size])
        windows = np.array(windows)

        # scale the windows (with the same scaler as used in training)
        if not (Path(models_dir)/'scalers_middle_nod.pkl').exists():
            print("No scaler found")
            return [], []

        scalers = joblib.load(Path(models_dir)/'scalers_middle_nod.pkl')
        for i in range(windows.shape[2]):
            windows[:, :, i] = scalers[i].transform(windows[:, :, i].reshape(-1, 1)).reshape(windows[:, :, i].shape)
            
        # show an example window
        if vector_i in to_show_pyr:
            plot_pyr_graphs(windows, number=1, diff_pix=False, diff_ang=False, movement_type='Unk')
                
        # Get the LSTM outputs for the middle frame
        predicted = model.predict(windows)
        predicted = np.argmax(predicted, axis=1).tolist()
        predicted = [2 if x == 1 else x for x in predicted]
        predictions.append(predicted)
            
    return labels, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('models_dir', type=Path)
    parser.add_arguments('window_size', type=int)
    parser.add_arguments('val_windows', nargs='+', type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    predict_lstm_class_windows_middle_nod(args.frames_csv, args.models_dir, args.window_size)

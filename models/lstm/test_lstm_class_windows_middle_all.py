import argparse
import pickle
from pathlib import Path
import math
import joblib

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from models.processing.facial_movement import derivatives_from_csv
from models.lstm.train_lstm_class_windows_middle_all import MODEL_LSTM_CLASS_FILENAME, PAD_VALUE, MASK_VALUE, plot_pyr_graphs
from models.simple.detector import verify_window_size
from utils.frames_csv import load_df, load_all_labels
from keras.models import Model

import tensorflow as tf

def predict_lstm_class_windows_middle_all(df_test, models_dir, window_size=36, load_data=False, random_windows=False, only_nods=True):
    window_size = verify_window_size(window_size)
    
    # load model
    model = tf.keras.models.load_model(models_dir/MODEL_LSTM_CLASS_FILENAME)
    
    # initialize predictions and labels
    predictions, labels = [], []

    # load data
    if (Path(models_dir)/'val_x_windows_middle_all.npy').exists() and (Path(models_dir)/'val_y_windows_middle_all.npy').exists() and load_data:
        
        # THIS DOESNT WORK
        # because it's random windows not ordered by video or anything
        
        print("Loading data from nmpys")
        val_x = np.load(Path(models_dir)/'val_x_windows_middle_all.npy')
        val_y = np.load(Path(models_dir)/'val_y_windows_middle_all.npy')

        # show an example window
        plot_pyr_graphs(val_x, number=3, add_diff=True, diff_pix=False, diff_ang=False, movement_type='Unk')

        # Get the LSTM outputs for the middle frame
        val_x = val_x.tolist()
        predictions = model.predict(val_x)
        predictions = np.argmax(predictions, axis=1).tolist()
        labels = val_y.tolist()
        conf_matr = confusion_matrix(labels, predictions)
        from validation.cross_validation import plot_confusion_matrix
        plot_confusion_matrix(conf_matr, title=None, labels = ['background', 'shake', 'nod'])

    else:
        print("Creating data")
        df_test = load_df(df_test)
        vectors = derivatives_from_csv(df_test).copy()
        labels = load_all_labels(df_test, shift=0, window=window_size)

        if only_nods:
            print("Removing files that do not contain nods")
            indexes_with_nods = [index for index, array in enumerate(labels) if 1 in array]
            print("Nr of files in validation data that contain nods: ", len(indexes_with_nods), "/", len(labels), ", namely the indexes: ", indexes_with_nods)
            labels = [labels[index] for index in indexes_with_nods]
            vectors = [vectors[index] for index in indexes_with_nods]

        if not random_windows:
            print("Not random windows")
            # for each video
            for vector in vectors:

                # make windows empty again
                windows = []

                # get nr of predictions
                pred_len = len(vector) - window_size + 1

                # Make windows of (masked/padded) length
                for i in range(pred_len):
                    windows.append(vector[i:i + window_size])

                # make windows start at 0
                windows = np.array(windows)
                # for w_i, w in enumerate(windows):
                #     for i in range(w.shape[1]):
                #         windows[w_i, :, i] = add_diff(windows[w_i, :, i])

                # scale the windows (with the same scaler as used in training)
                if not (Path(models_dir)/'scalers_middle_all.pkl').exists():
                    print("No scaler found")
                    return [], []

                scalers = joblib.load(Path(models_dir)/'scalers_middle_all.pkl')
                for i in range(windows.shape[2]):
                    windows[:, :, i] = scalers[i].transform(windows[:, :, i].reshape(-1, 1)).reshape(windows[:, :, i].shape)
                    
                # show an example window
                # if predictions == []:
                #     plot_pyr_graphs(windows, number=1, add_diff=False, diff_pix=False, diff_ang=False, movement_type='Unk')
                    
                # Get the LSTM outputs for the middle frame
                predicted = model.predict(windows)
                predicted = np.argmax(predicted, axis=1)
                predictions.append(predicted)
        else:
            labels = np.concatenate(labels).astype(int).tolist()
            new_labels = []
            for l in labels:
                if l == 0:
                    new_labels.append(0)
                elif l == 1:
                    new_labels.append(1)
                elif l == 2:
                    new_labels.append(2)
                else:
                    print("Label not recognized")
            labels = new_labels

            # make windows
            windows = []

            # for each video
            for vector in vectors:

                # get nr of predictions
                pred_len = len(vector) - window_size + 1

                # Make windows of (masked/padded) length
                for i in range(pred_len):
                    windows.append(vector[i:i + window_size])
            
            # make windows start at 0
            windows = np.array(windows)
            # for w_i, w in enumerate(windows):
            #     for i in range(w.shape[1]):
            #         windows[w_i, :, i] = add_diff(windows[w_i, :, i])

            # scale the windows (with the same scaler as used in training)
            if not (Path(models_dir)/'scalers_middle_all.pkl').exists():
                print("No scaler found")
                return [], []

            scalers = joblib.load(Path(models_dir)/'scalers_middle_all.pkl')
            for i in range(windows.shape[2]):
                windows[:, :, i] = scalers[i].transform(windows[:, :, i].reshape(-1, 1)).reshape(windows[:, :, i].shape)
                
            # show an example window
            # plot_pyr_graphs(windows, number=1, diff_pix=False, diff_ang=False, movement_type='Unk')
                    
            # Get the LSTM outputs for the middle frame
            predicted = model.predict(windows)
            predicted = np.argmax(predicted, axis=1).tolist()
            predictions = predicted
            
    return labels, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('models_dir', type=Path)
    parser.add_arguments('window_size', type=int)
    parser.add_arguments('val_windows', nargs='+', type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    predict_lstm_class_windows_middle_all(args.frames_csv, args.models_dir, args.window_size)

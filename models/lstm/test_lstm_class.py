import argparse
import pickle
from pathlib import Path
import math

import numpy as np
from sklearn.metrics import confusion_matrix

from models.processing.facial_movement import derivatives_from_csv
from models.processing.preparation import add_diff, add_diffs_list
from models.lstm.train_lstm_class import MODEL_LSTM_CLASS_FILENAME, PAD_VALUE, MASK_VALUE, plot_pyr_graphs
from models.simple.detector import verify_window_size
from utils.frames_csv import load_df, load_all_labels
from keras.models import Model

import tensorflow as tf

def predict_lstm_class(df_test, models_dir, window_size=36, val_windows=[35]):
    window_size = verify_window_size(window_size)

    df_test = load_df(df_test)
    vectors = derivatives_from_csv(df_test).copy()
    labels = load_all_labels(df_test, shift=0, window=window_size)

    # load model
    model = tf.keras.models.load_model(models_dir/MODEL_LSTM_CLASS_FILENAME)
    lstm_output_model = Model(inputs=model.input, outputs=model.layers[1].output)

    predictions = []

    # for each video
    for vector in vectors:

        # get nr of predictions
        pred_len = len(vector) - window_size + 1
        
        # make predictions for different window sizes
        preds, probs =[], []
            
        # for each window size
        for val_window in val_windows:

            # Make windows of (masked/padded) length
            windows = []
            for i in range(pred_len):
                window = vector[i:i + window_size].copy()
                if val_window < window_size:
                    # Divide by 2 and round down, e.g. (9-4)/5 -> 2
                    start = math.ceil((window_size - val_window) / 2)
                    # Mask out at the beginning and ending of window
                    window[0 : start] = [PAD_VALUE] * start
                    window[start:start + val_window] = add_diffs_list(window[start:start + val_window])
                    window[start + val_window : window_size] = [window[start + val_window - 1]] * (window_size - val_window - start)
                windows.append(window)

            # show an example window
            # if predictions == []:
            #     plot_pyr_graphs(windows, number=2, diff_pix=False, diff_ang=False, movement_type='Unk')

            
            # Get the LSTM outputs for the middle frame
            middle_frame_index = math.floor(window_size/2)
            middle_frame_input = windows[:, middle_frame_index, :].reshape(-1, 1, window_size)
            # make predictions for these windows
            predicted = model.predict(np.array(middle_frame_input))
            if len(val_windows) == 1:
                preds = np.argmax(predicted, axis=1)
            else:
                pred_frame, prob_frame = [], []
                for pr_frame in predicted:
                    index = np.where(pr_frame == np.max(pr_frame))[0][0]
                    prob_frame.append(pr_frame[index])
                    pred_frame.append(index)
                preds.append(pred_frame)
                probs.append(prob_frame)

        cur_max_probs, cur_max_preds = [], []
        if len(val_windows) == 1:
            cur_max_preds = preds
        else:
            for prob_i, prob in enumerate(probs):

                if prob_i == 0:
                    cur_max_probs = prob
                    cur_max_preds = [0]*len(cur_max_probs)
                else:
                    for i, p in enumerate(prob):
                        if p > cur_max_probs[i]:
                            cur_max_probs[i] = p
                            cur_max_preds[i] = preds[prob_i][i]
        predictions.append(cur_max_preds)
    return labels, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_csv', type=Path)
    parser.add_argument('models_dir', type=Path)
    parser.add_arguments('window_size', type=int)
    parser.add_arguments('val_windows', nargs='+', type=int)
    parser.add_argument('-o', '--overwrite', action='store_true')
    args = parser.parse_args()

    predict_lstm_class(args.frames_csv, args.models_dir, args.window_size)

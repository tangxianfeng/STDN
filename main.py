import models
import file_loader

import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
import sys
# import xgboost as xgb
import json
import gc
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
if len(sys.argv) == 3:
    gpu_num = sys.argv[2][2]
    if gpu_num != "0" and gpu_num != "1":
        print("\nError gpu code {0}, use default.".format(gpu_num))
        gpu_num = "0"
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    print("\n***** Selecting gpu {0}".format(gpu_num))
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from keras import backend as K
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Activation, concatenate, Input, Conv2D, Reshape, Flatten, Dropout, BatchNormalization, Concatenate
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, Callback, ModelCheckpoint
import datetime
import ipdb
import gc
import attention
from attention import Attention


class CustomStopper(keras.callbacks.EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch = 40):
        super().__init__(monitor = monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

def eval_together(y, pred_y, threshold):
    mask = y > threshold
    if np.sum(mask)==0:
        return -1
    mape = np.mean(np.abs(y[mask]-pred_y[mask])/y[mask])
    rmse = np.sqrt(np.mean(np.square(y[mask]-pred_y[mask])))

    return rmse, mape

def eval_lstm(y, pred_y, threshold):
    pickup_y = y[:, 0]
    dropoff_y = y[:, 1]
    pickup_pred_y = pred_y[:, 0]
    dropoff_pred_y = pred_y[:, 1]
    pickup_mask = pickup_y > threshold
    dropoff_mask = dropoff_y > threshold
    #pickup part
    if np.sum(pickup_mask)!=0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])/pickup_y[pickup_mask])
        avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[pickup_mask]-pickup_pred_y[pickup_mask])))
    #dropoff part
    if np.sum(dropoff_mask)!=0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])/dropoff_y[dropoff_mask])
        avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[dropoff_mask]-dropoff_pred_y[dropoff_mask])))

    return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)

def main(att_lstm_num = 3, long_term_lstm_seq_len = 3, short_term_lstm_seq_len = 7, cnn_nbhd_size = 3, nbhd_size = 2, cnn_flat_size = 128,\
     batch_size = 64, max_epochs = 100, validation_split = 0.2, early_stop = EarlyStopping()):
    model_hdf5_path = "./hdf5s/"

    model_name = ""
    if len(sys.argv) == 1:
        print("no parameters")
        return
    else:
        model_name = sys.argv[1]

    sampler = file_loader.file_loader()
    modeler = models.models()

    if model_name[2:] == "stdn":
        #training
        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype = "train", att_lstm_num = att_lstm_num,\
        long_term_lstm_seq_len = long_term_lstm_seq_len, short_term_lstm_seq_len = short_term_lstm_seq_len,\
        nbhd_size = nbhd_size, cnn_nbhd_size = cnn_nbhd_size)

        print("Start training {0} with input shape {2} / {1}".format(model_name[2:], x.shape, cnnx[0].shape))

        model = modeler.stdn(att_lstm_num = att_lstm_num, att_lstm_seq_len = long_term_lstm_seq_len,\
        lstm_seq_len = len(cnnx), feature_vec_len = x.shape[-1],\
        cnn_flat_size = cnn_flat_size, nbhd_size = cnnx[0].shape[1], nbhd_type = cnnx[0].shape[-1])


        model.fit(\
        x = att_cnnx + att_flow + att_x + cnnx + flow + [x,],\
        y = y,\
        batch_size=batch_size, validation_split = validation_split, epochs=max_epochs, callbacks=[early_stop])

        att_cnnx, att_flow, att_x, cnnx, flow, x, y = sampler.sample_stdn(datatype = "test", nbhd_size = nbhd_size, cnn_nbhd_size = cnn_nbhd_size)
        y_pred = model.predict(\
        x = att_cnnx + att_flow + att_x + cnnx + flow + [x,],)
        threshold = float(sampler.threshold) / sampler.config["volume_train_max"]
        print("Evaluating threshold: {0}.".format(threshold))
        (prmse, pmape), (drmse, dmape) = eval_lstm(y, y_pred, threshold)
        print("Test on model {0}:\npickup rmse = {1}, pickup mape = {2}%\ndropoff rmse = {3}, dropoff mape = {4}%".format(model_name[2:], prmse, pmape*100, drmse, dmape*100))
        
        currTime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        model.save(model_hdf5_path + model_name[2:] + currTime + ".hdf5")
        return

    print("Cannot recognize parameter...")
    return

stop = CustomStopper(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='min', start_epoch = 40)
batch_size = 64
max_epochs = 1000

if __name__ == "__main__":
    main(batch_size = batch_size, max_epochs = max_epochs, early_stop = stop)
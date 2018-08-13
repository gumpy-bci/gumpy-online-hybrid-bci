# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
# 03.03´4.2018
# class used for live EEG with a CNN for classification based on CNN-py by Sai Lam Loo


from __future__ import print_function
import sys

sys.path.append('..\..')

import numpy as np
import gumpy
from gumpy.data.nst_eeg_live import NST_EEG_LIVE

#import scipy.io
from scipy.signal import decimate #,butter, lfilter, spectrogram
#import matplotlib.pyplot as plt
import keras
#from keras.utils import plot_model
#from sklearn.model_selection import train_test_split

#from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Conv2D, MaxPooling2D #,LSTM
import keras.utils as ku
from keras.callbacks import ModelCheckpoint, CSVLogger

import kapre
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D
#from kapre.augmentation import AdditiveNoise
from datetime import datetime
import os
import os.path

DEBUG = 1

def check_model(model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])
    batch_input_shape = (2,) + model.input_shape[1:]
    batch_output_shape = (2,) + model.output_shape[1:]
    model.compile('sgd', 'mse')
    model.fit(np.random.uniform(size=batch_input_shape), np.random.uniform(size=batch_output_shape), epochs=1)



###############################################################################
#def load_model(model_directory, model_file_name, weights_file_name):
#    #TODO: does not work, but is not required
#    try:
#        # load trained model
#        model_path = model_file_name + ".json"
#        if not os.path.isfile(model_path):
#            raise IOError('file "%s" does not exist' % (model_path))
#        model = model_from_json(open(model_path).read(),custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram})
#
#        # load weights of trained model
#        model_weight_path = weights_file_name + ".hdf5"
#        if not os.path.isfile(model_path):
#            raise OSError('file "%s" does not exist' % (model_path))
#        model.load_weights(model_weight_path)
#
#        return model
#    except IOError:
#        print(IOError)
#        return None



###############################################################################
class liveEEG_CNN():
    def __init__(self, data_dir, filename_notlive, n_classes = 2):
        self.print_version_info()

        self.data_dir = data_dir
        self.cwd = os.getcwd()
        self.n_classes = n_classes
        kwargs = {'n_classes': self.n_classes}

        ### initialise dataset
        self.data_notlive = NST_EEG_LIVE(self.data_dir, filename_notlive,**kwargs)
        self.data_notlive.load()
        self.data_notlive.print_stats()

        self.MODELNAME = "CNN_STFT"

        self.x_stacked = np.zeros((1, self.data_notlive.sampling_freq*self.data_notlive.trial_total, 3))
        self.y_stacked = np.zeros((1, self.n_classes))

        self.fs = 256
        self.lowcut = 2
        self.highcut = 60
        self.anti_drift = 0.5
        self.f0 = 50.0  # freq to be removed from signal (Hz) for notch filter
        self.Q = 30.0  # quality factor for notch filter
        # w0 = f0 / (fs / 2)
        self.AXIS = 0
        self.CUTOFF = 50.0
        self.w0 = self.CUTOFF / (self.fs / 2)
        self.dropout = 0.5

        ### reduce sampling frequency to 256
        ### most previous data is at 256 Hz, but no it has to be recorded at 512 Hz due to the combination of EMG and EEG
        ### hence, EEG is downsampled by a factor of 2 here
        if self.data_notlive.sampling_freq > self.fs:
            self.data_notlive.raw_data = decimate(self.data_notlive.raw_data, int(self.data_notlive.sampling_freq/self.fs), axis=0, zero_phase=True)
            self.data_notlive.sampling_freq = self.fs
            self.data_notlive.trials = np.floor(self.data_notlive.trials /2).astype(int)

        ### filter the data
        self.data_notlive_filt = gumpy.signal.notch(self.data_notlive.raw_data, self.CUTOFF, self.AXIS)
        self.data_notlive_filt = gumpy.signal.butter_highpass(self.data_notlive_filt, self.anti_drift, self.AXIS)
        self.data_notlive_filt = gumpy.signal.butter_bandpass(self.data_notlive_filt, self.lowcut, self.highcut, self.AXIS)

        #self.min_cols = np.min(self.data_notlive_filt, axis=0)
        #self.max_cols = np.max(self.data_notlive_filt, axis=0)

        ### clip and normalise the data
        ### keep normalisation constants for lateron (hence no use of gumpy possible)
        self.sigma = np.min(np.std(self.data_notlive_filt, axis=0))
        self.data_notlive_clip = np.clip(self.data_notlive_filt, self.sigma * (-6), self.sigma * 6)

        self.notlive_mean = np.mean(self.data_notlive_clip, axis=0)
        self.notlive_std_dev = np.std(self.data_notlive_clip, axis=0)
        self.data_notlive_clip = (self.data_notlive_clip-self.notlive_mean)/self.notlive_std_dev
        #self.data_notlive_clip = gumpy.signal.normalize(self.data_notlive_clip, 'mean_std')

        ### extract the time within the trials of 10s for each class
        self.class1_mat, self.class2_mat = gumpy.utils.extract_trials_corrJB(self.data_notlive, filtered = self.data_notlive_clip)#, self.data_notlive.trials,
                                                    #self.data_notlive.labels, self.data_notlive.trial_total, self.fs)#, nbClasses=self.n_classes)
        #TODO: correct function extract_trials() trial len & trial offset

        ### concatenate data for training and create labels
        self.x_train = np.concatenate((self.class1_mat, self.class2_mat))
        self.labels_c1 = np.zeros((self.class1_mat.shape[0],))
        self.labels_c2 = np.ones((self.class2_mat.shape[0],))
        self.y_train = np.concatenate((self.labels_c1, self.labels_c2))

        ### for categorical crossentropy as an output of the CNN, another format of y is required
        self.y_train = ku.to_categorical(self.y_train)

        if DEBUG:
            print("Shape of x_train: ", self.x_train.shape)
            print("Shape of y_train: ", self.y_train.shape)

        print("EEG Data loaded and processed successfully!")

        ### roll shape to match to the CNN
        self.x_rolled = np.rollaxis(self.x_train, 2, 1)

        if DEBUG:
            print('X shape: ', self.x_train.shape)
            print('X rolled shape: ', self.x_rolled.shape)

        ### augment data to have more samples for training
        self.x_augmented, self.y_augmented = gumpy.signal.sliding_window(data=self.x_train,
                                                          labels=self.y_train, window_sz=4*self.fs, n_hop=self.fs//8, n_start=self.fs*3)

        ### roll shape to match to the CNN
        self.x_augmented_rolled = np.rollaxis(self.x_augmented, 2, 1)
        print("Shape of x_augmented: ", self.x_augmented_rolled.shape)
        print("Shape of y_augmented: ", self.y_augmented.shape)


        ### try to load the .json model file, otherwise build a new model
        self.loaded = 0
        if os.path.isfile(os.path.join(self.cwd,self.MODELNAME+".json")):
            self.load_CNN_model()
            if self.model:
                self.loaded = 1

        if self.loaded == 0:
            print("Could not load model, will build model.")
            self.build_CNN_model()
            if self.model:
                self.loaded = 1

        ### Create callbacks for saving
        saved_model_name = self.MODELNAME
        TMP_NAME = self.MODELNAME + "_" + "_C" + str(self.n_classes)
        for i in range(99):
            if os.path.isfile(saved_model_name + ".csv"):
                saved_model_name = TMP_NAME + "_run{0}".format(i)

        ### Save model -> json file
        json_string = self.model.to_json()
        model_file = saved_model_name + ".json"
        open(model_file, 'w').write(json_string)

        ### define where to save the parameters to
        model_file = saved_model_name + 'monitoring' + '.h5'
        checkpoint = ModelCheckpoint(model_file, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        log_file = saved_model_name + '.csv'
        csv_logger = CSVLogger(log_file, append=True, separator=';')
        self.callbacks_list = [csv_logger, checkpoint]  # callback list



###############################################################################
    ### train the model with the notlive data or sinmply load a pretrained model
    def fit(self, load=False):
        #TODO: use method train_on_batch() to update model
        self.batch_size = 32
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

        if not load:
            print('Train...')
            self.model.fit(self.x_augmented_rolled, self.y_augmented,
                 batch_size=self.batch_size,
                 epochs=100,
                 shuffle=True,
                 validation_split=0.2,
                 callbacks=self.callbacks_list)
        else:
            print('Load...')
            self.model = keras.models.load_model('CNN_STFTmonitoring.h5',
                                     custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram,
                                     'Normalization2D': kapre.utils.Normalization2D})

        #CNN_STFT__C2_run4monitoring.h5

###############################################################################
    ### do the live classification
    def classify_live(self, data_live):
        ### perform the same preprocessing steps as in __init__()

        ### agina, donwsampling from 512 to 256 (see above)
        if data_live.sampling_freq > self.fs:
            data_live.raw_data = decimate(data_live.raw_data, int(self.data_notlive.sampling_freq/self.fs), axis=0, zero_phase=True)
            data_live.sampling_freq = self.fs

        self.y_live=data_live.labels

        self.data_live_filt = gumpy.signal.notch(data_live, self.CUTOFF, self.AXIS)
        self.data_live_filt = gumpy.signal.butter_highpass(self.data_live_filt, self.anti_drift, self.AXIS)
        self.data_live_filt = gumpy.signal.butter_bandpass(self.data_live_filt, self.lowcut, self.highcut, self.AXIS)

        self.data_live_clip = np.clip(self.data_live_filt, self.sigma * (-6), self.sigma * 6)
        self.data_live_clip = (self.data_live_clip-self.notlive_mean)/self.notlive_std_dev

        class1_mat, class2_mat = gumpy.utils.extract_trials_corrJB(data_live, filtered=self.data_live_clip)

        ### concatenate data  and create labels
        self.x_live = np.concatenate((class1_mat, class2_mat))

        self.x_live = self.x_live[:,
                    data_live.mi_interval[0]*data_live.sampling_freq\
                    :data_live.mi_interval[1]*data_live.sampling_freq, :]

        self.x_live = np.rollaxis(self.x_live, 2, 1)

        ### do the prediction
        pred_valid = 0
        y_pred = []
        pred_true = []
        if self.loaded and self.x_live.any():
            y_pred = self.model.predict(self.x_live,batch_size=64)
            print(y_pred)
            #classes = self.model.predict(self.x_live_augmented,batch_size=64)
            #pref0 = sum(classes[:,0])
            #pref1 = sum(classes[:,1])
            #if pref1 > pref0:
            #    y_pred = 1
            #else:
            #    y_pred = 0

            ### argmax because output is crossentropy
            y_pred = y_pred.argmax()
            pred_true = self.y_live == y_pred
            print('Real=',self.y_live)
            pred_valid = 1

        return  y_pred, pred_true, pred_valid



###############################################################################
    def load_CNN_model(self):
        print('Load model', self.MODELNAME)
        model_path = self.MODELNAME + ".json"
        if not os.path.isfile(model_path):
            raise IOError('file "%s" does not exist' % (model_path))
        self.model = model_from_json(open(model_path).read(),custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram,
                                     'Normalization2D': kapre.utils.Normalization2D})
        #self.model = load_model(self.cwd,self.MODELNAME,self.MODELNAME+'monitoring')
        #TODO: get it to work, but not urgently required
        #self.model = []



###############################################################################
    def build_CNN_model(self):
        ### define CNN architecture
        print('Build model...')
        self.model = Sequential()
        self.model.add(Spectrogram(n_dft=128, n_hop=16, input_shape=(self.x_augmented_rolled.shape[1:]),
                              return_decibel_spectrogram=False, power_spectrogram=2.0,
                              trainable_kernel=False, name='static_stft'))
        self.model.add(Normalization2D(str_axis = 'freq'))

        # Conv Block 1
        self.model.add(Conv2D(filters = 24, kernel_size = (12, 12),
                         strides = (1, 1), name = 'conv1',
                         border_mode = 'same'))
        self.model.add(BatchNormalization(axis = 1))
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2), padding = 'valid',
                               data_format = 'channels_last'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # Conv Block 2
        self.model.add(Conv2D(filters = 48, kernel_size = (8, 8),
                         name = 'conv2', border_mode = 'same'))
        self.model.add(BatchNormalization(axis = 1))
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'valid',
                               data_format = 'channels_last'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # Conv Block 3
        self.model.add(Conv2D(filters = 96, kernel_size = (4, 4),
                         name = 'conv3', border_mode = 'same'))
        self.model.add(BatchNormalization(axis = 1))
        self.model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2),
                               padding = 'valid',
                               data_format = 'channels_last'))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(self.dropout))

        # classificator
        self.model.add(Flatten())
        self.model.add(Dense(self.n_classes))  # two classes only
        self.model.add(Activation('softmax'))

        print(self.model.summary())
        self.saved_model_name = self.MODELNAME



###############################################################################
    def print_version_info(self):
        now = datetime.now()

        print('%s/%s/%s' % (now.year, now.month, now.day))
        print('Keras version: {}'.format(keras.__version__))
        if keras.backend._BACKEND == 'tensorflow':
            import tensorflow
            print('Keras backend: {}: {}'.format(keras.backend._backend, tensorflow.__version__))
        else:
            import theano
            print('Keras backend: {}: {}'.format(keras.backend._backend, theano.__version__))
        print('Keras image dim ordering: {}'.format(keras.backend.image_dim_ordering()))
        print('Kapre version: {}'.format(kapre.__version__))



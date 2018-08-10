from __future__ import print_function
import sys

sys.path.append('./scripts')

import numpy as np
import scipy.io
from scipy.signal import butter, lfilter, spectrogram
import matplotlib.pyplot as plt
import data_processing
import keras
from keras.utils import plot_model
from sklearn.model_selection import train_test_split

from keras.preprocessing import sequence
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Flatten, BatchNormalization, LSTM, Dropout
import keras.utils as ku
from keras.callbacks import ModelCheckpoint, CSVLogger
import kapre
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D
from kapre.augmentation import AdditiveNoise
from datetime import datetime
import os.path
#from ggplot import *  # for beautiful plots
import pylsl
now = datetime.now()


def print_version_info():
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



def check_model(model):
    model.summary(line_length=80, positions=[.33, .65, .8, 1.])
    batch_input_shape = (2,) + model.input_shape[1:]
    batch_output_shape = (2,) + model.output_shape[1:]
    model.compile('sgd', 'mse')
    model.fit(np.random.uniform(size=batch_input_shape), np.random.uniform(size=batch_output_shape), epochs=1)


def load_model(model_directory, model_file_name):
    try:
        # load trained model
        model_path = model_file_name + ".json"
        if not os.path.isfile(model_path):
            raise IOError('file "%s" does not exist' % (model_path))
        model = model_from_json(open(model_path).read())

        # load weights of trained model
        model_weight_path = model_file + ".hdf5"
        if not os.path.isfile(model_path):
            raise OSError('file "%s" does not exist' % (model_path))
        model.load_weights(model_weight_path)

        return model
    except IOError:
        print(IOError)
        return None

#####the code above is just some import


################get the EEG filtered signal from the Windows.
if __name__ == '__main__':
    print_version_info()

    print('Resolving streams')
    stream = pylsl.resolve_stream('type','EEGPre') # EEG for the raw signal; EEGPre for a preprocessed signal
    print('Found at least one stream')
    inlet = pylsl.stream_inlet(stream[0])
    info = inlet.info()
    print(info.as_xml())

    while True:
    	
        x_streamed, time = inlet.pull_sample()
        X = np.array(x_streamed)
        X_test = np.zeros((1,3,1024))
        X_test[0,0,:]= X[:1024]
        X_test[0, 1, :] = X[1024:2048]
        X_test[0, 2, :] = X[2048:3072]
        print(X_test.shape)
            
    
        ####################I get the EEG signals two 4s-signal of 3 channels for one 10s-trails.
        ###################
    
    
    
        ##########this is your classifier I use the threshold. you'd better look for the value of a ##########and b to change the threshold.
    
    
    
        model2 = keras.models.load_model('CNN_STFT__C2_run4monitoring.h5',custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram})
    	
        classes = model2.predict(X_test)
    
        a = classes[:, 0]
        print(a)
        b = classes[:, 1]
        print(b)
        c = a/b
        if (c>4):
        	os.system("./control katana6M180.cfg 192.168.1.4 left")
        elif(0.5<c<4):
    
        	os.system("./control katana6M180.cfg 192.168.1.4 null")
        else:
    	    os.system("./control katana6M180.cfg 192.168.1.4 right")
    

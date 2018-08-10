import math
import numpy as np
import processing_filters as pf
#from scipy.misc import imresize # requires Pillow
from scipy.signal import spectrogram

class filterbank:

	def __init__(self, lowcut=2, highcut=60, order=3, fs=256):
		self.bandPass = pf.butter_bandpass(lowcut,highcut,order,fs)
		self.notch = pf.butter_bandstop()
		#notch = pf.notch()

	def process(self, data):
		buf = self.bandPass.process(data)
		return self.notch.process(buf)

class specGen:

    def __init__(self, width = 32, height = 32, numChannels = 3, lowf = 2, periods = 1.5, overlapRatio = 0.95, fs=256):
        self.width = width
        self.height = height
        self.channels = numChannels
        self.fs = fs
        self.SFFTwindowWidth = int(math.ceil(fs/lowf * periods))
        self.SFFToverlap = int(math.floor(self.SFFTwindowWidth * overlapRatio))
        self.smpPerSpec = int(self.SFFTwindowWidth + (self.width - 1) * (self.SFFTwindowWidth - self.SFFToverlap))

    def process(self, data):
        #for iChannel in xrange(self.channels)
        # spec_X: [Y, X] = [f, t]
        f,t,spec_1 = spectrogram(data[:, 0], self.fs, nperseg=self.SFFTwindowWidth, noverlap=self.SFFToverlap, detrend=False)#[2]
        spec_2 = spectrogram(data[:, 1], self.fs, nperseg=self.SFFTwindowWidth, noverlap=self.SFFToverlap, detrend=False)[2]
        spec_3 = spectrogram(data[:, 2], self.fs, nperseg=self.SFFTwindowWidth, noverlap=self.SFFToverlap, detrend=False)[2]
        specs = np.zeros((self.channels, self.height, self.width))
        if spec_1.shape[1]>self.width:
            start = spec_1.shape[1] - self.width
        else:
            start = 0
        specs[0, :, :] = spec_1[:self.height, start:].copy()
        specs[1, :, :] = spec_2[:self.height, start:].copy()
        specs[2, :, :] = spec_3[:self.height, start:].copy()
        # specs[0, :, :] = imresize(arr=spec_1, size=(self.height, self.width), interp='nearest', mode='F')
        # specs[1, :, :] = imresize(arr=spec_2, size=(self.height, self.width), interp='nearest', mode='F')
        # specs[2, :, :] = imresize(arr=spec_3, size=(self.height, self.width), interp='nearest', mode='F')
        # print f
        return specs

def np_to_tn(input_data):
	# normailzes data and casts to uint8 for TN usage
	tmp = np.array(normalize_3D(input_data))
	out_255 = tmp * 255
	out_norm = out_255.astype('uint8')
	return  out_norm

def normalize_3D(x):
	x_min = x.min(axis=(0, 1), keepdims=True)
	x_max = x.max(axis=(0, 1), keepdims=True)
	x = (x - x_min)/(x_max-x_min)
	return x
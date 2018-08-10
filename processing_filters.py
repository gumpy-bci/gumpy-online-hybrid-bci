"""Collection of Butterworth IIR filter classes"""

from scipy.signal import butter, filtfilt, iirnotch

__author__ = "Christian Widderich"

class butter_lowpass:

	def __init__(self, cutoff, order, fs=256):
		self.cutoff = cutoff
		self.order = order

		nyq = 0.5 * fs
		low = cutoff / nyq
		self.b, self.a = butter(order, low, btype='lowpass')

	def process(self, data, axis=0):
		return filtfilt(self.b, self.a, data, axis)

class butter_highpass:

	def __init__(self, cutoff, order, fs=256):
		self.cutoff = cutoff
		self.order = order

		nyq = 0.5 * fs
		high = cutoff / nyq
		self.b, self.a = butter(order, high, btype='highpass')

	def process(self, data, axis=0):
		return filtfilt(self.b, self.a, data, axis)

class butter_bandpass:

	def __init__(self, lowcut, highcut, order, fs=256):
		self.lowcut = lowcut
		self.highcut = highcut
		self.order = order

		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq
		self.b, self.a = butter(order, [low, high], btype='bandpass')

	def process(self, data, axis=0):
		return filtfilt(self.b, self.a, data, axis)

class butter_bandstop:

	def __init__(self, lowpass=49, highpass=51, order=4, fs=256):
		self.lowpass = lowpass
		self.highpass = highpass
		self.order = order

		nyq = 0.5 * fs
		low = lowpass / nyq
		high = highpass / nyq
		self.b, self.a = butter(order, [low, high], btype='bandstop')

	def process(self, data, axis=0):
		return filtfilt(self.b, self.a, data, axis)

class notch:

	def __init__(self, cutoff=50, Q=30, fs=256):
		self.cutoff = cutoff
		self.Q = Q

		nyq = 0.5 * fs
		cut = cutoff / nyq
		self.b, self.a = iirnotch(cut, Q)

	def process(self, data, axis=0):
		return filtfilt(self.b, self.a, data, axis)
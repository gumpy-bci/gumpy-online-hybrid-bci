# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
# 27.03.2018
# based on: record_data_liveEMG.py and record_data_liveEEG.py to allow live processing and classification of EEG with a CNN
# record_data_liveEEG.py  by:  Mirjam Hemberger, mirjam.hemberger@tum.de
# function record_and_process() based on realtime.py by whoever
# function control() based on real_time.py by whoever

import os, sys

if 0: #os.name == "nt":
    # DIRTY workaround from stackoverflow
    # when using scipy, a keyboard interrupt will kill python
    # so nothing after catching the keyboard interrupt will
    # be executed

    #import imp
    #import ctypes
    import _thread
    import win32api

    #basepath = imp.find_module('numpy')[1]
    #print(basepath)
    #ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
    #ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

    def handler(dwCtrlType, hook_sigint=_thread.interrupt_main):
        if dwCtrlType == 0:
            hook_sigint()
            return 1
        return 0

    win32api.SetConsoleCtrlHandler(handler, 1)


import threading           # NOQA
import scipy.io as sio     # NOQA
import pylsl               # NOQA
from utils import time_str # NOQA
import time                # NOQA
#import serial
import math
import numpy as np
#from scipy.signal import spectrogram


#from matplotlib import pyplot as plt
#import matplotlib.gridspec as gridspec


import ringbuffer
import live_processing as lp
#from nst_emg_live_180301 import NST_EMG_LIVE
from gumpy.data.nst_eeg_live import NST_EEG_LIVE

import keras
#from keras.models import load_model
import kapre
from real_time_copy import print_version_info #, check_model, load_model
#sys.path.append('./scripts')
#import data_processing

class NoRecordingDataError(Exception):
    def __init__(self):
        self.value = "Received no data while recording"

    def __str__(self):
        return repr(self.value)



def find_gUSBamp_stream(name):
    # required, because two different amplifiers, both sending a stream with type 'EEG' can be connected
    # the one with the label EMG will be used for EMG and the one with label EEG for EEG
    # returns the correct stream
    streams = pylsl.resolve_stream('type', 'EEG')
    if name == 'EEG':
        device_id = 'g.USBamp-UB-2016.08.03'
        n_channels = 9
    elif name == 'EMG':
        device_id = 'g.USBamp-UB-2016.08.04'
        n_channels = 8
    inlet = []
    for i in range(len(streams)):
        try:
            if streams[i].name() == device_id and streams[i].channel_count() == n_channels:
                inlet = pylsl.stream_inlet(streams[i])
        except:
           pass
    return inlet



def record(stop_event, channel_data=[],time_stamps=[]):
    # this is the old recording function, that is started as a thread in the class RecordData
    inlet   = find_gUSBamp_stream('EEG')
    inletInfo = inlet.info()
    print('Connected to:',inletInfo.name(), 'with', inletInfo.channel_count(),'channels. Fs:',inletInfo.nominal_srate())
    # do recording all the time, but to pause the recording a stop event is set.
    # agterwards the thread is newly initialised to restart
    while not stop_event.is_set(): #True:
        try:
            sample, time_stamp = inlet.pull_sample()
            time_stamp += inlet.time_correction()

            time_stamps.append(time_stamp)
            channel_data.append(sample)


        except KeyboardInterrupt:
            # save data and exit on KeybordInterrupt
            complete_samples = min(len(time_stamps), len(channel_data))
            sio.savemat("recording_" + time_str() + ".mat", {
                "time_stamps"  : time_stamps[:complete_samples],
                "channel_data" : channel_data[:complete_samples],
            })
            break



def record_and_process(stop_event, channel_data=[], time_stamps=[]):
    # based on realtime.py
    # 0. General
    #verbose = False
    # 1. Output
    output_width    = 32
    output_height   = 32
    output_stacks   = 3  # channels
    outlet_sendRate = 2 # [Hz]
    outlet_numChannels = output_width*output_height*output_stacks
    # 2. Filterbank
    lowcut  = 2  # [Hz]
    highcut = 60 # [Hz]
    order   = 3
    # 3. Spectrogram Generation
    periods      = 1.5
    overlapRatio = 0.95

    # initialise data inlet and outlet
    inlet   = find_gUSBamp_stream('EEG')
    inletInfo = inlet.info()

    inlet_sampleRate = int(inletInfo.nominal_srate())
    inlet_numChannels = int(inletInfo.channel_count())
    print("Reported sample rate: %i , number of channels: %i" %(inlet_sampleRate, inlet_numChannels))

    outletInfo    = pylsl.StreamInfo('PreprocessedEEG', 'EEGPre', outlet_numChannels, outlet_sendRate, 'int8', 'UB-2016.08.03')
    outlet = pylsl.StreamOutlet(outletInfo)

    # initialise processing: Filterbank, Spectrogram generation
    filterbank = lp.filterbank(lowcut,highcut,order,inlet_sampleRate)
    specGen = lp.specGen(output_width, output_height, output_stacks, lowcut, periods, overlapRatio, inlet_sampleRate)

    # initialise ringbuffer
    rbuffer = ringbuffer.RingBuffer(size_max=specGen.smpPerSpec)
    sendEverySmpl = math.ceil(inlet_sampleRate / outlet_sendRate)
    print("Transmitting every %i samples" %sendEverySmpl)

    samplesInBuffer = 0
    samplesSent = 0

    while not stop_event.is_set():
        try:

            sample, time_stamp = inlet.pull_sample()
            time_stamp += inlet.time_correction()

            time_stamps.append(time_stamp)
            channel_data.append(sample)
            rbuffer.append(sample)
            samplesInBuffer += 1
        except KeyboardInterrupt:
            # save data and exit on KeybordInterrupt
            complete_samples = min(len(time_stamps), len(channel_data))
            sio.savemat("recording_" + time_str() + ".mat", {
                "time_stamps"  : time_stamps[:complete_samples],
                "channel_data" : channel_data[:complete_samples],
            })
            break

        if(rbuffer.full and samplesInBuffer>=sendEverySmpl):
            # get from buffer, filter and generate spectrogram
            specs = specGen.process(filterbank.process(np.array(rbuffer.get())[:,0:3]) )
            # convert to uint8 and flatten to send it to the LSL
            outlet.push_sample(lp.np_to_tn(specs).flatten())
            samplesSent += 1
            # indicate that buffer content is used, such that it can be overwritten
            samplesInBuffer = 0



def control(stop_event, X=[]):
    # based on real_time.py
    stream = pylsl.resolve_stream('type','EEGPre') # EEG for the raw signal; EEGPre for a preprocessed signal
    inlet = pylsl.stream_inlet(stream[0])
    #info = inlet.info()
    print_version_info()

    while not stop_event.is_set():
        x_streamed, time = inlet.pull_sample()
        X = np.array(x_streamed)
        X_test = np.zeros((1,3,1024))
        X_test[0,0,:]= X[:1024]
        X_test[0, 1, :] = X[1024:2048]
        X_test[0, 2, :] = X[2048:3072]
        model2 = keras.models.load_model('CNN_STFT__C2_run7monitoring.h5',custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram})

        classes = model2.predict(X_test)
        a = classes[:, 0]
        print(a)
        b = classes[:, 1]
        print(b)
        c = a/b
        if (c>4):
            print('left')
            #move robot arm left
        elif(0.5<c<4):
            print('middle')
            #move robot arm to the middle
        else:
            print('right')
            #move robot arm to the right



class RecordData_liveEEG_JB():
    def __init__(self, Fs, age, gender="male", with_feedback=False,
                 record_func=record_and_process,control_func=control):

        # decide whether control thread should also be started
        # currently not used
        self.docontrol = False

        # indizes in X indicating the beginning of a new trial
        self.trial = []
        # list including all the recorded EEG data
        self.X = []

        # time stamp indicating the beginning of each trial
        self.trial_time_stamps = []
        # all time stamps, one is added for each data point
        self.time_stamps       = []
        # label of each trial: 0: left, 1: right, 2: both
        self.Y = []
        # currently not used # 0 negative_feedback # 1 positive feedback
        self.feedbacks = []
        # sampling frequency
        self.Fs = Fs
        # trial offset in motor imagery paradigm. Used in get_last_trial()
        self.trial_offset = 4

        self.gender   = gender
        self.age      = age
        self.add_info = "with feedback" if with_feedback else "with no feedback"

        # initialise a subfolder where the data is to be saved
        # if it does not yet exist, create it
        self.datapath = os.path.join(os.getcwd(),'00_DATA')
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)

        # stop event used to pause and resume to the recording & processing thread
        self.stop_event_rec = threading.Event()
        self.stop_event_con = threading.Event()

        # initialise recording thread. It does not run yet. Therefore use start_recording()
        recording_thread = threading.Thread(
            target=record_func,
            args=(self.stop_event_rec, self.X, self.time_stamps),
        )
        recording_thread.daemon = True
        self.recording_thread   = recording_thread

        # initialise control thread.
        if self.docontrol:
            control_thread = threading.Thread(
                    target=control_func,
                    args=(self.stop_event_con, self.X),
                    )
            control_thread.daemon = True
            self.control_thread   = control_thread


    def __iter__(self):
        yield 'trial'            , self.trial
        yield 'age'              , self.age
        yield 'X'                , self.X
        yield 'time_stamps'      , self.time_stamps
        yield 'trial_time_stamps', self.trial_time_stamps
        yield 'Y'                , self.Y
        yield 'Fs'               , self.Fs
        yield 'gender'           , self.gender
        yield 'add_info'         , self.add_info
        yield 'feedbacks'        , self.feedbacks



    def add_trial(self, label):
        # called whenever a new trial is started
        self.trial_time_stamps.append(pylsl.local_clock())
        self.Y.append(label)
        # trial includes the index in X and force, where each trial has begun
        self.trial.append(len(self.X)-1)



    def add_feedback(self, feedback):
        self.feedbacks.append(feedback)



    def start_recording(self,len_X = 0, len_f = 0):
        # start the recording thread
        self.recording_thread.start()

        if self.docontrol:
            self.control_thread.start()

        time.sleep(2)
        # check whether data arrived, if not raise error
        if len(self.X)-len_X == 0:
            raise NoRecordingDataError()



    def pause_recording(self):
        # raise stop_event to break the loop in record() while the classification of notlive data is done
        self.stop_event_rec.set()
        print('Recording has been paused.')



    def restart_recording(self):
        # newly initialise the recording thread and start it
        self.stop_event_rec.clear()
        recording_thread = threading.Thread(
            target=record,
            args=(self.stop_event_rec, self.X, self.time_stamps),
        )
        recording_thread.daemon = True
        self.recording_thread   = recording_thread
        self.start_recording(len_X=len(self.X))

        print('Recording has been restarted.')



    # this function is not required anymore, because self.trial is updated in add_trial()
    # kept for historical reasons
    def set_trial_start_indexes(self):
        # since it can be called twice during one recording (because of live processing)
        # everything done by the first step is deleted before the second step
        if len(self.trial) > 0:
            self.trial = []
        # the loop was once used to calculate the index in X that the time stamp of each trial begin relates to
        # this is solved by updating self.trial already in add_trial()
        i = 0
        for trial_time_stamp in self.trial_time_stamps:
            for j in range(i, len(self.time_stamps)):
                time_stamp = self.time_stamps[j]
                if trial_time_stamp <= time_stamp:
                    self.trial.append(j - 1)
                    i = j
                    break



    def stop_recording_and_dump(self, file_name="EEG_session_" + time_str() + ".mat"):
        # finish the recording, save all data to a .mat file
        self.pause_recording()
        self.stop_event_con.set()
        sio.savemat(os.path.join(self.datapath, file_name), dict(self))
        print('Recording will shut down.')
        return file_name, self.datapath



    def stop_recording_and_dump_live(self, file_name="EEG_session_live_" + time_str() + ".mat"):
        # still there for historic reasons, to support run_session by Mirjam Hemberger
        return self.continue_recording_and_dump()



    def continue_recording_and_dump(self, file_name="EEG_session_live_" + time_str() + ".mat"):
        # only save data while still keeping the recording thread alive
        # the data can then be used to classify the notlive data
        sio.savemat(os.path.join(self.datapath, file_name), dict(self))
        return file_name, self.datapath



    def pause_recording_and_dump(self, file_name="EEG_session_live_" + time_str() + ".mat"):
        # save data to .mat and pause the recording such that it can be resumed lateron
        sio.savemat(os.path.join(self.datapath, file_name), dict(self))
        self.pause_recording()
        return file_name, self.datapath



    def get_last_trial(self,filename_live = ""):
        # generate a NST_EEG_LIVE object and save data of last trial into it
        # the dataset can then be used for live classification
        last_label = self.Y[-1:]
        # subtract one trial offset, because add trial is allways called when the moto imagery starts and not in the beginning of each trial
        last_trial = self.trial[-1:][0]-self.Fs*self.trial_offset
        X = np.array(self.X[slice(last_trial,None,None)])
        if False: # usefull info for debugging
            print('last index is: ', last_trial[0])
            print('last time is:', self.trial_time_stamps[-1:][0])
            print('current time is: ', pylsl.local_clock())
            print('current index is:', len(self.X))

        # hand over 0 as index to dataset object, because the new index in the slice of X that will be handed over is 0
        last_trial = 0
        # generate an instance of the NST_EEG_LIVE class (inherits from Dataset class)
        self.nst_eeg_live = NST_EEG_LIVE(self.datapath, filename_live)
        # hand over data to the NST_EEG_LIVE instance
        self.nst_eeg_live.load_from_mat(last_label, last_trial, X, self.Fs)
        return self.nst_eeg_live



    def startAccumulate(self):
        self.accStart = len(self.X)
        print("starting accumulation")



    def stopAccumulate(self):
        pass


if __name__ == '__main__':
    pass #record()

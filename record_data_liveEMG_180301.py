# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
# 01.03.2018
# modification of record_data.py to allow live processing and classification of EMG




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
import serial
import numpy as np



#from nst_emg_live_180301 import NST_EMG_LIVE
import gumpy
from gumpy.data import NST_EMG_LIVE

class NoRecordingDataError(Exception):
    def __init__(self):
        self.value = "Received no data while recording"

    def __str__(self):
        return repr(self.value)



###############################################################################
def find_gUSBamp_stream(name):
    ### required, because two different amplifiers, both sending a stream with type 'EEG' can be connected
    ### the one with the label EMG will be used for EMG and the one with label EEG for EEG
    ### returns the correct stream
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



###############################################################################
def record(ser, stop_event, channel_data=[], channel_force=[], time_stamps=[]):
    ### this is the recording function, that is started as a thread in the class RecordData
    inlet = find_gUSBamp_stream('EMG')
    inletInfo = inlet.info()
    print('Connected to:',inletInfo.name(), 'with', inletInfo.channel_count(),'channels. Fs:',inletInfo.nominal_srate())
    
    #inlet2 = find_gUSBamp_stream('EEG')
    #inletInfo2 = inlet2.info()
    #print('Connected to:',inletInfo2.name(), 'with', inletInfo2.channel_count(),'channels. Fs:',inletInfo2.nominal_srate())
    
    ### do recording all the time, but to pause the recording a stop event is set.
    ### agterwards the thread is newly initialised to restart
    while not stop_event.is_set(): #True:
        try:
            sample, time_stamp = inlet.pull_sample()
            time_stamp += inlet.time_correction()
            
            time_stamps.append(time_stamp)
            channel_data.append(sample)
            if ser:
                ### if no arduino for force recording is connected, then fill with 0
                try:
                    channel_force.append(float(ser.readline()))   
                except ValueError:
                    channel_force.append(0)
                except TypeError:
                    channel_force.append(0)
            else:
                channel_force.append(0)
                                    
        except KeyboardInterrupt:
            ### save data and exit on KeybordInterrupt
            complete_samples = min(len(time_stamps), len(channel_data), len(channel_force))
            sio.savemat("recording_" + time_str() + ".mat", {
                "time_stamps"  : time_stamps[:complete_samples],
                "channel_data" : channel_data[:complete_samples],
                "channel_force" : channel_force[:complete_samples]
            })
            break



###############################################################################
class RecordData_liveEMG():
    def __init__(self, Fs, age, gender="male", with_feedback=False,
                 record_func=record):
        
        ### indizes in X and force matrix indicating the beginning of a new trial
        self.trial = []
        ### list including all the recorded EMG data
        self.X = []
        ### list including the recorded force data (or zeros)
        self.force = []
        ### time stamp indicating the beginning of each trial
        self.trial_time_stamps = []
        ### all time stamps, one is added for each data point
        self.time_stamps       = []
        ### label of each trial: 0: fist, 1: pinch_2, 2: pinch_3, +10: weak force
        self.Y = []
        ### currently not used # 0 negative_feedback # 1 positive feedback
        self.feedbacks = []
        ### sampling frequency
        self.Fs = Fs

        self.gender   = gender
        self.age      = age
        self.add_info = "with feedback" if with_feedback else "with no feedback"
        
        ### initialise a subfolder where the data is to be saved
        ### if it does not yet exist, create it
        self.datapath = os.path.join(os.getcwd(),'00_DATA')
        if not os.path.exists(self.datapath):
            os.makedirs(self.datapath)
        
        ### initialise serial port. ona different computer, this might be another one
        ### e.g. COM4. To check, go to the arduino program and it will show up in the bottom right corner
        ### if initialisatoin does not work, then we'll continue without recording the force (only zeros)
        port = "COM3"
        baudrate = 2000000
        try:
            self.serialPort = serial.Serial(port, baudrate) 
            self.serialPort.readline()
        except:
            print('Could not connect to Arduino, will generate "0" data for force.\n')
            self.serialPort = False
        
        ### stop event used to pause and resume to the recording thread
        self.stop_event = threading.Event()
        
        ### initialise recording thread. It does not run yet. Therefore use start_recording()
        recording_thread = threading.Thread(
            target=record_func,
            args=(self.serialPort, self.stop_event, self.X, self.force, self.time_stamps),
        )
        recording_thread.daemon = True
        self.recording_thread   = recording_thread



###############################################################################        
    def __iter__(self):
        yield 'trial'            , self.trial
        yield 'age'              , self.age
        yield 'X'                , self.X
        yield 'force'            , self.force
        yield 'time_stamps'      , self.time_stamps
        yield 'trial_time_stamps', self.trial_time_stamps
        yield 'Y'                , self.Y
        yield 'Fs'               , self.Fs
        yield 'gender'           , self.gender
        yield 'add_info'         , self.add_info
        yield 'feedbacks'        , self.feedbacks
        
   

###############################################################################    
    def add_trial(self, label):
        ### called whenever a new trial is started
        self.trial_time_stamps.append(pylsl.local_clock())
        self.Y.append(label)
        ### trial includes the index in X and force, where each trial has begun
        self.trial.append(len(self.X)-1)



###############################################################################
    def add_feedback(self, feedback):
        self.feedbacks.append(feedback)



###############################################################################
    def start_recording(self,len_X = 0, len_f = 0):
        ### start the recording thread
        self.recording_thread.start()

        time.sleep(2)
        ### check whether data arrived, if not raise error
        if len(self.X)-len_X == 0 or len(self.force)-len_f == 0:
            if len(self.X)-len_X == 0:
                print('No EMG data.', len(self.force))
            if len(self.force)-len_f == 0:
                print('No force data.', len(self.X))
            raise NoRecordingDataError()



###############################################################################            
    def pause_recording(self):
        ### raise stop_event to break the loop in record() while the classification of notlive data is done
        self.stop_event.set()
        print('Recording has been paused.')



###############################################################################            
    def restart_recording(self):
        ### newly initialise the recording thread and start it
        self.stop_event.clear()
        recording_thread = threading.Thread(
            target=record,
            args=(self.serialPort, self.stop_event, self.X, self.force, self.time_stamps),
        )
        recording_thread.daemon = True
        self.recording_thread   = recording_thread
        self.start_recording(len_X=len(self.X),len_f=len(self.force))
        
        print('Recording has been restarted.')
        


###############################################################################
    ### this function is not required anymore, because self.trial is updated in add_trial()
    ### kept for historical reasons
    def set_trial_start_indexes(self):
        ### since it can be called twice during one recording (because of live processing) 
        ### everything done by the first step is deleted before the second step
        if len(self.trial) > 0:
            self.trial = []
        ### the loop was once used to calculate the index in X that the time stamp of each trial begin relates to
        ### this is solved by updating self.trial already in add_trial()
        i = 0
        for trial_time_stamp in self.trial_time_stamps:
            for j in range(i, len(self.time_stamps)):
                time_stamp = self.time_stamps[j]
                if trial_time_stamp <= time_stamp:
                    self.trial.append(j - 1)
                    i = j
                    break



###############################################################################
    def stop_recording_and_dump(self, file_name="EMG_session_" + time_str() + ".mat"):
        ### finish the recording, save all data to a .mat file, close serial port
        self.pause_recording()
        sio.savemat(os.path.join(self.datapath, file_name), dict(self))
        print('Recording will shut down.')
        if self.serialPort:
            self.serialPort.close()
            
        return file_name, self.datapath



###############################################################################
    def continue_recording_and_dump(self, file_name="EMG_session_live_" + time_str() + ".mat"):
        ### only save data while still keeping the recording thread alive
        ### the data can then be used to classify the notlive data

        sio.savemat(os.path.join(self.datapath, file_name), dict(self))
        
        return file_name, self.datapath
    


###############################################################################    
    def pause_recording_and_dump(self, file_name="EMG_session_live_" + time_str() + ".mat"):
        ### save data to .mat and pause the recording such that it can be resumed lateron
        sio.savemat(os.path.join(self.datapath, file_name), dict(self))
        self.pause_recording()
        return file_name, self.datapath



###############################################################################
    def get_last_trial(self):
        ### generate a NST_EMG_LIVE object and save data of last trial into it
        ### the dataset can then be used for live classification
        last_label = self.Y[-1:]       
        last_trial = self.trial[-1:]
        X = np.array(self.X[slice(last_trial[0],None,None)])
        forces = np.array([self.force[slice(last_trial[0],None,None)]])
        if False: ### usefull info for debugging
            print('last index is: ', last_trial[0])
            print('last time is:', self.trial_time_stamps[-1:][0])
            print('current time is: ', pylsl.local_clock())
            print('current index is:', len(self.X))
            print(forces.shape)
        ### hand over 0 as index to dataset object, because the new index in the slice of X that will be handed over is 0
        last_trial = 0 
        ### generate an instance of the NST_EMG_LIVE class (inherits from Dataset class)
        self.nst_emg_live = NST_EMG_LIVE('','','')
        ### hand over data to the NST_EMG_LIVE instance
        self.nst_emg_live.load_from_mat(last_label,last_trial, X, forces, self.Fs)

        return self.nst_emg_live



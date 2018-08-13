# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
# 01.03.2018
# class used for liveEMG based on EMG-script.ipynb


#import matplotlib.pyplot as plt 

import sys #, os, os.path


import numpy as np
import gumpy
from gumpy.data import NST_EMG_LIVE
#import math
#import copy

#from nst_emg_live_180301 import NST_EMG_LIVE




class liveEMG():
    def __init__(self, cwd, filename_notlive):
        ### cwd and filename_notlive specify the location of one .mat file where the recorded notlive data has been stored
        
        ### load notlive data from path that has been specified
        self.data_notlive = NST_EMG_LIVE(cwd, filename_notlive, '')

        self.data_notlive.load()

        self.data_notlive.print_stats()
        print('----------')

        ### filter notlive data
        self.bp_low = 20
        self.bp_high = 255
        self.notch_f0 = 50
        self.notch_Q = 50       

        self.data_notlive_filt = gumpy.signal.butter_bandpass(self.data_notlive, self.bp_low, self.bp_high)
        self.data_notlive_filt = gumpy.signal.notch(self.data_notlive_filt, cutoff=self.notch_f0, Q=self.notch_Q)
        
        ### extract trials from notlive data
        self.trials_notlive = gumpy.utils.getTrials(self.data_notlive, self.data_notlive_filt)

        ### extract RMS features
        self.window_size = 0.2
        self.window_shift = 0.05

        self.RMS_features_notlive = self.RMS_features_extraction(self.data_notlive, self.trials_notlive)
        
        ### prepare arrays for classification
        self.X_class_notlive = self.RMS_features_notlive
        self.X_norm = np.linalg.norm(self.X_class_notlive)
        
        # exclude the background posture, because it might make problems
        # 0,1,2 is postures for high force // 10,11,12 are postures for low force
        # -> modulo 10 to get the posture type
        # -> 10*floor(label/10) to get force type (either 0 or 10)
        self.y_class_notlive = self.data_notlive.labels[:self.X_class_notlive.shape[0]] %10
        self.y_class_force = 10*np.floor(self.data_notlive.labels[:self.X_class_notlive.shape[0]]/10)
                   
        self.X_class_notlive = np.divide(self.X_class_notlive,self.X_norm,
                                         out=np.zeros_like(self.X_class_notlive),
                                         where=self.X_norm!=0)
        self.X_class_notlive_norm = np.zeros(1)
                        
        self.pos_fit = False
        self.force_fit = False


    
###############################################################################    
    def fit(self, classifier='NaiveBayes', X=[],y=[],split = 0):
        ### if X and y are not entered, then the ones generated in __init()__ will be used
        ### if they are entered, X should be a matrix of size [N_trials, N_features]
        ### and y a matrix of size [N_trials] including the classifier for each trial
        if X != []:
            self.X_class_notlive = X
        if y != []:
            self.y_class_notlive = y % 10
            self.y_class_force = 10 * np.floor(y/10)
        
        ### choose default classifier if wrong classifier
        if not classifier in gumpy.classification.available_classifiers:
            self.classifier = 'NaiveBayes'
        else:
            self.classifier = classifier            
        print('**********',self.classifier,'**********')
        
        ### used for testing procedure.
        ### only uses the first N trials of the dataset. N = split
        if split != 0:
            print('Only first',split,'trials of each posture will be used.')
            self.split_dataset(split)
  
        ### if force classification has been selected then fit force as well.
        ### otherwise skip this part
        if np.sum(self.y_class_force) >= 10:
            print("Classify Force and Posture.\n")
            
            ### if the force will also be classified, then normalise the two different force levels diferently
            ### we know that first half of trials is high force and second half is low force
            ### and put them back together
            N2 = int(np.floor(self.X_class_notlive.shape[0]/2))
            self.X_norm_high = 1
            self.X_norm_low = np.divide(np.linalg.norm(self.X_class_notlive[N2:]),np.linalg.norm(self.X_class_notlive[:N2]))
            print('high:',self.X_norm_high, 'low:', self.X_norm_low)
            self.X_class_notlive_norm = np.concatenate((
                np.divide(self.X_class_notlive[:N2],self.X_norm_high,
                     out=np.zeros_like(self.X_class_notlive[:N2]),where=self.X_norm_high!=0),
                np.divide(self.X_class_notlive[N2:],self.X_norm_low,
                     out=np.zeros_like(self.X_class_notlive[N2:]),where=self.X_norm_low!=0)))
            
            
            ### perform force level classification with sequential feature selection
            out_force_realtime = gumpy.features.sequential_feature_selector_realtime \
                (self.X_class_notlive, self.y_class_force, self.classifier, (1,3), 3, 'SFFS',False)
            self.sfs_object_force = out_force_realtime[3]
            self.estimator_object_force = self.sfs_object_force.est_
            ### fit the estimator object with the selected features and test
            self.estimator_object_force.fit(self.sfs_object_force.transform(self.X_class_notlive), self.y_class_force)
            self.y_pred_notlive_force = self.estimator_object_force.predict(self.sfs_object_force.transform(self.X_class_notlive))
            self.acc_notlive_force = 100 - 100 * np.sum(abs(self.y_pred_notlive_force-self.y_class_force)>=1) \
                / np.shape(self.y_pred_notlive_force)
            print('\nAccuracy of notlive force fit:', self.acc_notlive_force[0], '\n')            
            self.force_fit = True
            
        else:
            print("Classify Posture only.\n")
            
        ### perform classification with sequential feature selection
        ### if force is also fit and the force accuracy was good, 
        ### then use the features then normalise amplitude according to force level
        ### otherwise just do normal feature selection
        if self.X_class_notlive_norm.shape[0] > 1 and self.acc_notlive_force > 75:
            out_post_realtime = gumpy.features.sequential_feature_selector_realtime \
                (self.X_class_notlive_norm, self.y_class_notlive, self.classifier, (5,10), 3, 'SFFS',False) 
            self.sffs_score = out_post_realtime[1]
            self.sfs_object = out_post_realtime[3]
            self.estimator_object = self.sfs_object.est_

            ### fit the estimator object with the selected features and test
            self.estimator_object.fit(self.sfs_object.transform(self.X_class_notlive_norm), self.y_class_notlive)
            self.y_pred_notlive = self.estimator_object.predict(self.sfs_object.transform(self.X_class_notlive_norm))
            
        else:
            out_post_realtime = gumpy.features.sequential_feature_selector_realtime \
                (self.X_class_notlive, self.y_class_notlive, self.classifier, (5,10), 3, 'SFFS',False) 
            #self.sfs_index = out_post_realtime[0]
            self.sffs_score = out_post_realtime[1]
            self.sfs_object = out_post_realtime[3]
            self.estimator_object = self.sfs_object.est_

            ### fit the estimator object with the selected features and test
            self.estimator_object.fit(self.sfs_object.transform(self.X_class_notlive), self.y_class_notlive)
            self.y_pred_notlive = self.estimator_object.predict(self.sfs_object.transform(self.X_class_notlive))
                
        self.acc_notlive = 100 - 100 * np.sum(abs(self.y_pred_notlive-self.y_class_notlive)>=1) \
                / np.shape(self.y_pred_notlive)
        print('\nAccuracy of notlive posture fit:', self.acc_notlive[0], '\n')
        self.pos_fit = True
        
        ### if split has been performed, do a test on the remaining data
        if hasattr(self, 'X_test_notlive'): # if dataset has been split, now evaluate the remaining data
            if self.X_test_notlive.shape[0]:
                y_pred_notlive_test = self.estimator_object.predict(self.sfs_object.transform(self.X_test_notlive))
                if np.sum(self.y_class_force) >= 10:
                    y_pred_notlive_test += self.estimator_object_force.predict(self.sfs_object.transform(self.X_test_notlive))
                acc_test = 100 - 100 * np.sum(abs(y_pred_notlive_test-self.y_test_notlive)>=1) \
                    / np.shape(y_pred_notlive_test)
                print('\nAccuracy of test fit after split:', acc_test[0], '\n')



###############################################################################
    def classify_live(self, data_live):
        ### data_live should be an object of class NST_EMG_LIVE 
        ### do the same preprocessing steps as for notlive data
        
        if not isinstance(data_live, NST_EMG_LIVE):
            print('data_live has to be an instance of NST_EMG_LIVE')
            return False, False, False
        
        ### filter live data
        try:
            data_live_filt = gumpy.signal.butter_bandpass(data_live, self.bp_low, self.bp_high)
            data_live_filt = gumpy.signal.notch(data_live_filt, cutoff=self.notch_f0, Q=self.notch_Q)
        except ValueError:
            print('ValueError: data too small')
            return False, False, False
        
        ### extract trials from live data
        trials_live = gumpy.utils.getTrials(data_live, data_live_filt)

        ### extract RMS features
        RMS_features_live = self.RMS_features_extraction(data_live, trials_live)

        ### prepare arrays for classification
        X_class_live = RMS_features_live #np.vstack((, RMS_features_live_Bg))

        ### don't divide by norm of the trial but by norm of the trining data set
        X_class_live = np.divide(X_class_live,self.X_norm,
                                         out=np.zeros_like(X_class_live),
                                         where=self.X_norm!=0)
        y_class_live = data_live.labels

        ### predict label of live trial and check whether it is correct
        if self.force_fit and self.pos_fit:
            print("Actual posture:", y_class_live)
            y_pred_live = self.estimator_object_force.predict(self.sfs_object_force.transform(X_class_live))

            ### use knowledge about force level to normalise the data differently for posture classification
            if y_pred_live == 0:
                X_class_live = np.divide(X_class_live,self.X_norm_high,
                                         out=np.zeros_like(X_class_live),
                                         where=self.X_norm_high!=0)
            elif y_pred_live == 10:
                X_class_live = np.divide(X_class_live,self.X_norm_low,
                                         out=np.zeros_like(X_class_live),
                                         where=self.X_norm_low!=0)
                
            y_pred_live += self.estimator_object.predict(self.sfs_object.transform(X_class_live))

        elif self.pos_fit:
            y_pred_live = self.estimator_object.predict(self.sfs_object.transform(X_class_live))
        
        elif self.force_fit:
            y_pred_live = self.estimator_object_force.predict(self.sfs_object_force.transform(X_class_live))
        
        else:
            print("No fit was performed yet. Please train the model first.\n")
        
        pred_true = (y_class_live - y_pred_live) == 0
        pred_valid = True
        return y_pred_live, pred_true, pred_valid



###############################################################################
    ### this function calculates the RMS features within the period  when the posture is performed
    def RMS_features_extraction(self, data, trialList):
                
        if self.window_shift > self.window_size:
            raise ValueError("window_shift > window_size")

        fs = data.sampling_freq
    
        n_features = int(data.duration/(self.window_size-self.window_shift))
    
        X = np.zeros((len(trialList), n_features*4))
    
        t = 0
        for trial in trialList:
            x1=gumpy.signal.rms(trial[0], fs, self.window_size, self.window_shift)
            x2=gumpy.signal.rms(trial[1], fs, self.window_size, self.window_shift)
            x3=gumpy.signal.rms(trial[2], fs, self.window_size, self.window_shift)
            x4=gumpy.signal.rms(trial[3], fs, self.window_size, self.window_shift)
            x=np.concatenate((x1, x2, x3, x4))
            try:
                X[t, :] = np.array([x])
            except:
                print(t)
                print(np.array([x]).shape)
                print(X.shape)
            t += 1
        return X    



###############################################################################
    ### this function takes only the first N trials for training, with N=split
    ### it can be used for testing the effect of training data size on testing
    def split_dataset(self, split):
        
        self.indizes = []
        y_copy = self.y_class_force+self.y_class_notlive
        X_copy = self.X_class_notlive[:]
        
        classes_possible = np.unique(y_copy)
        
        for pos in classes_possible:
            for i in range(split):
                for index in range(len(y_copy)):
                    if y_copy[index] == pos:
                        y_copy[index] = float('nan')
                        self.indizes.append(index)
                        break
                    
        y_copy = self.y_class_notlive[:]+self.y_class_force[:]                
        self.y_class_notlive = y_copy[self.indizes]%10
        self.y_class_force = 10*np.floor(y_copy[self.indizes]/10)
        self.X_class_notlive = X_copy[self.indizes,:]
        
        not_indizes = []
        for i in range(len(y_copy)):
            if i not in self.indizes:
                not_indizes.append(i)
      
        self.y_test_notlive = y_copy[not_indizes]
        self.X_test_notlive = X_copy[not_indizes,:]
        
        return self
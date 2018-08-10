# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
# 29.03.2018
# combination of run_session_liveEEG_JB.py and run_session_liveEMG.py
# to hybrid BCI experiments

### required for EMG
from math import pi, sin, cos
import os
import random
import sys
import argparse
#import time
import numpy as np


from direct.showbase.ShowBase import ShowBase
#from direct.task import Task
from direct.actor.Actor import Actor

from record_data_liveEMG_180301 import RecordData_liveEMG
from live_EMG_180301 import liveEMG
from robothand_arduino_180321 import RobotHand

### required for EEG
import pygame
import re
import time
import screeninfo
from record_data_liveEEG_JB import RecordData_liveEEG_JB
from live_EEG_CNN_180403 import liveEEG_CNN
### if not CNN but machine learning should be used
#from eeg_motor_imagery_NST_live import liveEEG
from robotarm_KUKA_180405 import RobotArm


on_windows = os.name == 'nt'

if on_windows:
    import winsound

    
    
class hybridBCI(ShowBase):
    def __init__(self, force_classify, trial_count,trials_notlive, mode, Fs, age, gender="male", with_feedback = False):
        ### copy all keybord inputs into object
        
        ### whether force classification for EMG will be used or not (0 or 1)
        self.force_classify = force_classify
        ### total number of trials in both offline and online phase
        self.trial_count = trial_count
        ### number of trials in the offline pahse before training the models
        self.trials_notlive = trials_notlive
        ### mode is EMG or EEG or HYBRID
        self.mode = mode
        ### sampling frequency in Hertz - has to be the same for EMG and EEg
        self.Fs = Fs
        ### age of the subject
        self.age = age
        ### gender of the subject
        self.gender = gender
        ### currently not used, but kept for history reasons
        self.with_feedback = with_feedback
        
        if self.mode == "EMG":
            self.init_EMG()
            ### some things of EEG need to be initialised because they are cnhecked as conditions in the state machine
            self.EEG_cue_pos_choices_notlive = []
            self.EEG_cue_pos_choices_live = []
            self.state = 1
            self.robotarmconnected = 0
        elif self.mode == "EEG":
            #required, because of task, but only reduced version will be executed
            self.init_EMG() 
            self.init_EEG()
            ### some things of EEG need to be initialised because they are cnhecked as conditions in the state machine
            self.EMG_cue_pos_choices_notlive = []
            self.EMG_cue_pos_choices_live = []
            self.robothandconnected = 0
            self.state = 11
        elif self.mode == "HYBRID":
            self.init_EMG()
            self.init_EEG()
            self.state = 11
        else:
            print("mode must either be EMG, EEG or HYOBRID.")
            self.state = 0
        if self.trial_count == 0:
            self.state = 0
        

###############################################################################        
    def init_EEG(self):
        self.EEG_pos_choices = ["left", "right", "centre"]
        
        ### init the pygame window with lots of commands
        self.screen_width, self.screen_height = self.get_screen_width_and_height()
        if 1: ###make sure it is not fullscreen, because if so the combination with EMG screen is difficult
            self.screen_width = int(self.screen_width*2/3)
            self.screen_height = int(self.screen_height*2/3)
        
        self.cwd = os.getcwd()
        self.datapath = self.cwd
        self.root_dir  = os.path.join(os.path.dirname(__file__), "..")
        self.image_dir = os.path.join(self.cwd, "images") #self.root_dir, 
        self.sound_dir = os.path.join(self.cwd, "sounds") #self.root_dir, 

        self.black   = (0,   0, 0)
        self.green   = (0, 255, 0)
        self.radius  = 100
        self.mid_pos = (self.screen_width // 2, self.screen_height // 2)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(os.path.join(self.sound_dir, "beep.mp3"))

        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.RESIZABLE)#     FULLSCREEN)
        self.screen.fill(self.black)
        
        ### initialise objects to be shown on the screen
        self.red_arrow       = pygame.image.load(os.path.join(self.image_dir, "red_arrow.png"))
        self.red_arrow_left  = pygame.transform.rotate(self.red_arrow, 270)
        self.red_arrow_right = pygame.transform.rotate(self.red_arrow, 90)

        self.red_arrow_width, self.red_arrow_height = self.red_arrow_left.get_size()
        self.red_arrow_right_pos = (self.screen_width - self.red_arrow_width, (self.screen_height - self.red_arrow_height) // 2)
        self.red_arrow_left_pos  = (0, (self.screen_height - self.red_arrow_height) // 2)

        self.happy_smiley = pygame.image.load(os.path.join(self.image_dir, "happy_smiley.png"))
        self.sad_smiley   = pygame.image.load(os.path.join(self.image_dir, "sad_smiley.png"))

        self.smiley_width, self.smiley_height = self.happy_smiley.get_size()
        self.smiley_mid_pos = ((self.screen_width - self.smiley_width) // 2, (self.screen_height - self.smiley_height) // 2)

        
        ### initialise variables, that are required lateron
        self.filename_notlive = ""
        self.current_classifier = []

        ### check trial length and initialise trials
        if self.trials_notlive > self.trial_count:
            raise ValueError("'trials_notlive' cannot be larger than 'trials'")
        
        if self.trial_count % 3:
            raise ValueError("'trials' must be devisable by 3")

        if self.trials_notlive % 3:
            raise ValueError("'trials_notlive' must be devisable by 3")

        self.EEG_trial_count_for_each_cue_pos = self.trial_count // 3
        self.EEG_trial_count_for_each_cue_pos_notlive = self.trials_notlive // 3

        self.EEG_cue_pos_choices_notlive = {
            "left"  : self.EEG_trial_count_for_each_cue_pos_notlive,
            "right" : self.EEG_trial_count_for_each_cue_pos_notlive,
            "both"  : self.EEG_trial_count_for_each_cue_pos_notlive
        }

        self.EEG_cue_pos_choices_live = {
            "left"  : (self.EEG_trial_count_for_each_cue_pos-self.EEG_trial_count_for_each_cue_pos_notlive),
            "right" : (self.EEG_trial_count_for_each_cue_pos-self.EEG_trial_count_for_each_cue_pos_notlive),
            "both"  : (self.EEG_trial_count_for_each_cue_pos-self.EEG_trial_count_for_each_cue_pos_notlive)
        }
        for cue_pos in ("left","right","both"):
            if self.EEG_cue_pos_choices_live[cue_pos] == 0:
                del self.EEG_cue_pos_choices_live[cue_pos]  
                
        #print(cue_pos_choices)
        print("\nNot live trials:")
        print(self.EEG_cue_pos_choices_notlive)
        print("\nLive trials:")
        print(self.EEG_cue_pos_choices_live, '\n')
        
        ### initialise recorddata object and start recordin
        self.record_data_EEG = RecordData_liveEEG_JB(self.Fs, self.age, self.gender, self.with_feedback)
        self.record_data_EEG.start_recording()
        
        ### try to connect to the KUKA robotic arm
        try:
            self.robotarm = RobotArm()
            self.robotarmconnected = True
        except:
            print('Robothand not connected. Output only to console')
            self.robotarmconnected = False
            
        ### initialise counters for online classification
        self.EEG_true = 0
        self.EEG_false = 0
        
        
###############################################################################        
    def init_EMG(self):
        ### call init of base class ShowBase
        ShowBase.__init__(self)
        
        ### this if condition includes all the commands that only have to be called when EMG is used
        ### the commands afterwards, i.e. all the init of Panda, have to be done for only EEG as well, because the taskmanager is used
        if self.mode == "EMG" or self.mode == "HYBRID":
            self.EMG_pos_choices = ["fist", "pinch_2", "pinch_3"]
            
            ### check inputs, whether the size is acceptable
            self.num_pos = len(self.EMG_pos_choices)
            if self.trials_notlive > self.trial_count:
                    raise ValueError("'trials_notlive' cannot be larger than 'trials'")
            
            
            if self.force_classify == True:
                # case when force shall be classified as well
                if self.trial_count % (2*self.num_pos):
                    raise ValueError("'trials' must be devisable by ", 2*self.num_pos)
    
                if self.trials_notlive % (2*self.num_pos):
                    raise ValueError("'trials_notlive' must be devisable by ", 2*self.num_pos)
    
                self.EMG_trial_count_for_each_cue_pos = self.trial_count // self.num_pos // 2
                self.EMG_trial_count_for_each_cue_pos_notlive = self.trials_notlive // self.num_pos // 2
                print("\nFirst, please apply strong force for ", self.trials_notlive // 2, " trials.\n", \
                      "Afterwards, please apply weak force for ", self.trials_notlive // 2, "trials.\n" )
            else:
                # noone cares about force, only posture
                if self.trial_count % self.num_pos:
                    raise ValueError("'trials' must be devisable by ", self.num_pos)
    
                if self.trials_notlive % self.num_pos:
                    raise ValueError("'trials_notlive' must be devisable by ", self.num_pos)
    
                self.EMG_trial_count_for_each_cue_pos = self.trial_count // self.num_pos
                self.EMG_trial_count_for_each_cue_pos_notlive = self.trials_notlive // self.num_pos
    
            ### Generate position lists for both live and notlive trials
                
            self.EMG_cue_pos_choices_notlive = [x for pair in zip(self.EMG_pos_choices*self.EMG_trial_count_for_each_cue_pos_notlive) for x in pair]
            self.EMG_cue_pos_choices_live = [x for pair in zip(self.EMG_pos_choices*(self.EMG_trial_count_for_each_cue_pos - 
                                                self.EMG_trial_count_for_each_cue_pos_notlive)) for x in pair]
    
            # Randomizing the positions
            random.shuffle(self.EMG_cue_pos_choices_notlive)
            random.shuffle(self.EMG_cue_pos_choices_live)
            
            if self.force_classify==True:
                # repeat the exact same testing procedure -->once strong and once weak
                self.EMG_cue_pos_choices_notlive = self.EMG_cue_pos_choices_notlive + self.EMG_cue_pos_choices_notlive #.append(self.cue_pos_choices_notlive)
                self.EMG_cue_pos_choices_live = self.EMG_cue_pos_choices_live + self.EMG_cue_pos_choices_live #.append(self.cue_pos_choices_live)
                self.EMG_trial_count_for_each_cue_pos = self.EMG_trial_count_for_each_cue_pos*2
                self.EMG_trial_count_for_each_cue_pos_notlive = self.EMG_trial_count_for_each_cue_pos_notlive*2
                
            print("\nNot live trials:")
            print(self.EMG_cue_pos_choices_notlive)
            print("\nLive trials:")
            print(self.EMG_cue_pos_choices_live, '\n')
            
    
            #Add of a position to avoid data loss, because pop(0) is used
            self.EMG_cue_pos_choices_notlive.append('end')
            self.EMG_cue_pos_choices_live.append('end')
    
            ### initialise the RecordData thread and the RobotHand
            self.record_data_EMG = RecordData_liveEMG(self.Fs, self.age, self.gender, with_feedback=False)
            self.record_data_EMG.start_recording()
            try:
                self.robothand = RobotHand(port="COM6") #"COM4" on Windows measurement PC
                self.robothandconnected = 1
            except:
                print('Robothand not connected. Output only to console')
                self.robothandconnected = 0
            
        ### initialise tasks, including Panda and the run_trial task
        ### this one is required for only EEG as well, because the taskmanager is used as well

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Add the run_trial procedure to the task manager
        self.taskMgr.add(self.run_trial, "run_trial_task")

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/Hand")

        scale = 10
        self.pandaActor.setScale(scale, scale, scale)
        self.pandaActor.reparentTo(self.render)

        self.pandaActor.setPos(7.9, 1.5, -14.5)

        ### init variables to be used lateron
        self.EMG_filename_notlive = ""
        self.cwd = os.getcwd()
        self.datapath = self.cwd
        self.EMG_current_classifier = []
        self.EMG_true = 0
        self.EMG_false = 0
        
        

###############################################################################        
        ### used for pygame
    def get_screen_width_and_height(self):
        monitor_info = screeninfo.get_monitors()[0]
        if not monitor_info:
            sys.exit("couldn't find monitor")
        m = re.match("monitor\((\d+)x(\d+)\+\d+\+\d+\)", str(monitor_info))

        self.screen_width, self.screen_height = int(m.group(1)), int(m.group(2))
        return self.screen_width, self.screen_height



###############################################################################
    def play_beep(self):
        pygame.mixer.music.play()    



###############################################################################
    ### Define a procedure to move the camera. Used for Panda animation
    def spinCameraTask(self, task):
        angleDegrees = 205#20 * task.time * 6.0
        theta = 20
        angleRadians = angleDegrees * (pi / 180.0)
        thetaRad = theta * (pi / 180.0)
        self.camera.setPos(3.5*sin(angleRadians), -3.5*cos(angleRadians), -3.5*sin(thetaRad))
        self.camera.setHpr(angleDegrees, theta, 0)

        return task.cont



###############################################################################
    # this decorator is defined to protect the classification function to run only once
    # otherwise the sequential feature selector causes problems, because it uses
    # the parallel processing toolbox
    def run_once(f):
        def wrapper(*args, **kwargs):
            if not wrapper.has_run:
                wrapper.has_run = True
                return f(*args, **kwargs)
        wrapper.has_run = False
        return wrapper



###############################################################################
    @run_once
    def classify_notlive(self):
    ### does the classification (called in state 2)
        if __name__ == '__main__':
            print('\nClassification is starting. This might take a while.\n')
            
            ### first do EEG if required
            if self.mode == "EEG" or self.mode == "HYBRID":
                ### pause recording while classification such that data file does get to large
                ### alternatively, recording can be continued, but in that case restart_recording() is not required
                self.filename_notlive, self.datapath = self.record_data_EEG.pause_recording_and_dump()
                print(self.datapath, ' ', self.filename_notlive, '\n')
                ### initialise liveEEG_CNN class with the notlive data and perform the fit of the model
                self.liveEEG = liveEEG_CNN(self.datapath, self.filename_notlive)
                #self.liveEEG = liveEEG(self.datapath, self.filename_notlive) ###if machine learning should be used
                self.liveEEG.fit(load=True)
                ### resume recording
                self.record_data_EEG.restart_recording()
            
            ### then do EMG 
            if self.mode == "EMG" or self.mode == "HYBRID":
                self.EMG_cue_pos_choices_notlive = ['end']
                ### pause recording while classification such that data file does get to large
                ### alternatively, recording can be continued, but in that case restart_recording() is not required       
                self.EMG_filename_notlive, self.datapath = self.record_data_EMG.pause_recording_and_dump()
                #self.filename_notlive = self.record_data.continue_recording_and_dump()
                print(self.datapath, '  ', self.EMG_filename_notlive, '\n')
                ### initialise liveEMG class with the not live data and perform the fit of the model
                self.liveEMG = liveEMG(self.datapath,self.EMG_filename_notlive)
                self.liveEMG.fit()
                ### resume recording
                self.record_data_EMG.restart_recording()
            
            print('Classification completed. Back in run_session.\n')
            


###############################################################################            
    def classify_live(self):
    ### called in state 4
    
        ### first do EEG if required
        if self.mode == "EEG" or self.mode == "HYBRID":
            filename_live, datapath = self.record_data_EEG.continue_recording_and_dump() 
            ### get_last_trial returns an instance of the Dataset class NST_EEG_LIVE, which is directly handed over to the liveEEG object
            self.EEG_current_classifier, pred_true, pred_valid = self.liveEEG.classify_live(self.record_data_EEG.get_last_trial(filename_live))
            if pred_valid != False:
                # display result of classification
                print('---------- EEG Classification result:',self.EEG_current_classifier)#, 
                      #self.EEG_pos_choices[self.EEG_current_classifier], '----------')
                if pred_true: 
                    print('---------- This is true! ----------\n')
                    self.EEG_true +=1
                else:
                    print('---------- This is false! ----------\n')
                    self.EEG_false +=1
                
                ### move the robotarm if it is connected
                if self.robotarmconnected:
                    self.robotarm.do_posture(self.EEG_current_classifier)
            else:
                print('This trial is skipped.\n') 
                
        ### then do EMG
        if self.mode == "EMG" or self.mode == "HYBRID":
            ### get_last_trial returns an instance of the Dataset class NST_EMG_LIVE, which is directly handed over to the liveEMG object
            self.EMG_current_classifier, pred_true, pred_valid = self.liveEMG.classify_live(self.record_data_EMG.get_last_trial())
            if pred_valid != False:
                # display result of classification
                print('---------- EMG Classification result: posture ',self.EMG_current_classifier, 
                      self.EMG_pos_choices[int(np.remainder(self.EMG_current_classifier,10))], '----------')
                if pred_true: 
                    print('---------- This is true! ----------\n')
                    self.EMG_true +=1
                else:
                    print('---------- This is false! ----------\n')
                    self.EMG_false +=1
                ### move the robothand if it is connected
                if self.robothandconnected:
                    self.robothand.do_posture(self.EMG_current_classifier,3)
     
            else:
                print('This trial is skipped because recording was too short.\n') 
        
        ### got back to initial position if robotic arm is connected
        if self.robotarmconnected:
                    self.robotarm.return_home()



###############################################################################            
    def run_notlive_EMG(self):
        ### state 1: record data nonlive
        ### state 1.5 is for applying low force 
        pos = self.EMG_cue_pos_choices_notlive.pop(0)
            
        if self.force_classify == True and  len(self.EMG_cue_pos_choices_notlive) == (self.EMG_trial_count_for_each_cue_pos_notlive*self.num_pos // 2):
            ### transintion from 1 to 1.5 after half of the notlive trials
            print("From now on, please apply low force!")
            self.state = 1.5
        ### do the hand movement
        self.pandaActor.play(pos)
        ### add_trial adds the timestamp and label to the recorded data for later identification of individual trials
        if self.state == 1:
            self.record_data_EMG.add_trial(self.EMG_pos_choices.index(pos))
        elif self.state == 1.5:
            ### for low force, 10 is added to all labels 
            self.record_data_EMG.add_trial(self.EMG_pos_choices.index(pos)+10)

        print('Now:',pos,'\t--- Next:',self.EMG_cue_pos_choices_notlive[0],'\t---',len(self.EMG_cue_pos_choices_notlive)-1,'left.')
    


###############################################################################    
    def run_live_EMG(self):
        # state 3: record data live
        # state 3.5 is for applying low force        
        pos = self.EMG_cue_pos_choices_live.pop(0)
        
        if self.force_classify == True and  len(self.EMG_cue_pos_choices_live) == \
            ((self.EMG_trial_count_for_each_cue_pos-self.EMG_trial_count_for_each_cue_pos_notlive)*self.num_pos // 2):
            ### transintion from 3 to 3.5 after half of the live trials
            print("From now on, please apply low force!")
            self.state = 3.5
        
        ### do the hand movement
        self.pandaActor.play(pos)
        ### add_trial adds the timestamp and label to the recorded data for later identification of individual trials
        if self.state == 3:
            self.record_data_EMG.add_trial(self.EMG_pos_choices.index(pos))
        elif self.state == 3.5:
            ### for low force, 10 is added to all labels 
            self.record_data_EMG.add_trial(self.EMG_pos_choices.index(pos)+10)

        print('Now:',pos,'\t--- Next:',self.EMG_cue_pos_choices_live[0],'\t---',len(self.EMG_cue_pos_choices_live)-1,'left.')
     
               
        
###############################################################################
        ### perform the experimental morot imagery paradigm as defined in the gumpy paper
    def show_motor_imagery(self, cue_pos_choices, with_feedback=False):
        ### t=0: black
        self.screen.fill(self.black)
        pygame.display.update()
        time.sleep(3)
        
        ### t=3: dot to focus
        pygame.draw.circle(self.screen, self.green, self.mid_pos, self.radius)
        pygame.display.update()
        time.sleep(1)
        
        ### t=4: display cue and add_trial()
        ### choosing a random cue
        cue_pos = random.choice(list(cue_pos_choices.keys()))
        cue_pos_choices[cue_pos] -= 1
        if cue_pos_choices[cue_pos] == 0:
            del cue_pos_choices[cue_pos]
        
        print("Now:\t", cue_pos, "\tRemaining:\t", cue_pos_choices)
        if cue_pos == "left":
            self.screen.blit(self.red_arrow_left, self.red_arrow_left_pos)
            self.record_data_EEG.add_trial(1)
        elif cue_pos == "right":
            self.screen.blit(self.red_arrow_right, self.red_arrow_right_pos)
            self.record_data_EEG.add_trial(2)
        elif cue_pos == "both":
            self.screen.blit(self.red_arrow_right, self.red_arrow_right_pos)
            self.screen.blit(self.red_arrow_left, self.red_arrow_left_pos)
            self.record_data_EEG.add_trial(3)
        pygame.display.update()
        
        ### t=4.5: play beep to close eyes
        time.sleep(0.5)

        if on_windows:
            #t = time.time()
            winsound.MessageBeep() #winsound.Beep(2500, 500)
            time.sleep(0.5)
            #print(time.time() - t)
            time.sleep(3)
        else:
            self.play_beep()
            time.sleep(3.5)
        
        ### t=8: display black, play beep to open eyes
        self.screen.fill(self.black)
        pygame.display.update()

        if on_windows:
            winsound.MessageBeep() #Beep(2500, 500)
            time.sleep(0.5)
            time.sleep(1.5)
        else:
            self.play_beep()
            time.sleep(2)

        if with_feedback:
            one_or_zero = random.choice([0, 1])
            smiley = [self.sad_smiley, self.happy_smiley][one_or_zero]
            self.record_data_EEG.add_feedback(one_or_zero)
            self.screen.blit(smiley, self.smiley_mid_pos)
            pygame.display.update()
            time.sleep(3)

        return cue_pos_choices
    
    
    
###############################################################################
    ### this is the function that actually runs all the time and performs the temporal sequence of states
    ### it has been added to the taskmanager
    ### this is the state-machine calling the different functions for each state
    def run_trial(self, task): 
        
        if task.time < 10.0: ### both playing the gesture in panda and performing the gesture with the robot arm take 10s
            return task.cont
        
        ### state 1 and 1.5: do notlive recording (1 for high force, 1.5 for low force)
        if self.state == 1 or self.state == 1.5:    
            self.run_notlive_EMG()
            
            ### change state from 1 to 11 or 1.5 to 11.5 if EEG offline needs to be done
            if len(self.EEG_cue_pos_choices_notlive) >=1:
                self.state +=10
                return task.again
            ### go to classification if offline EMG is done and online EMG is to come
            if len(self.EMG_cue_pos_choices_notlive) <= 1 and len(self.EMG_cue_pos_choices_live) >1:
                self.state = 2
            ### if EMG notlive remains, stay in state 1
            elif len(self.EMG_cue_pos_choices_notlive) > 1:
                return task.again
            ### if neither EMG nor EEG live remain, exit
            elif len(self.EMG_cue_pos_choices_live) <=1 and len(self.EEG_cue_pos_choices_live) < 1:
                self.state = 0
                            
            return task.again
        
        ### state 2: do classification of notlive data and change to state 3
        elif self.state == 2:         
            self.classify_notlive()
            ### if EEG live trials have to be done, go to EEG live otherwise to EMG live
            if len(self.EEG_cue_pos_choices_live) >= 1:
                self.state = 13
            else:
                self.state = 3
            return task.again
        
        ### state 3: do EMG live recording
        elif self.state == 3 or self.state == 3.5:                         
            ###  start next trial (only if there is one trial left. otherwise change state to 0 (end))
            if  len(self.EMG_cue_pos_choices_live) <=1 and len(self.EEG_cue_pos_choices_live) <1:
                self.state = 0
            ### if no EMG live trials remaining, go to EEG live
            elif len(self.EMG_cue_pos_choices_live) <=1:
                self.state += 10
            else:
                self.run_live_EMG()
                ### change to 4 or 4.5 (classify live), important to keep information of high or low
                self.state += 1
    
            return task.again
        
        ### state 4 do live classification and robot arm movement ->afterwards go back to state 3/3.5/13/13.5  
        elif self.state == 4 or self.state == 4.5:            
            self.classify_live()
            ### if EEG remaining, do EEG live before EMG live
            if len(self.EEG_cue_pos_choices_live) >= 1:
                self.state += 9
            ### change to 3 or 3.5 (EMG live), important to keep information of high or low
            elif len(self.EMG_cue_pos_choices_live) > 1:
                self.state -= 1
            ### no EMG or EEG live remaining --> exit
            else:
                self.state = 0
   
            return task.again
        
        ### notlive EEG, same principle as notlive EMG
        elif self.state == 11 or self.state == 11.5:
            if len(self.EEG_cue_pos_choices_notlive) >= 1:
                self.EEG_cue_pos_choices_notlive = self.show_motor_imagery(self.EEG_cue_pos_choices_notlive)
                
            if len(self.EMG_cue_pos_choices_notlive) > 1:
                self.state -=10
                return task.again
            
            if len(self.EEG_cue_pos_choices_notlive) < 1 and len(self.EEG_cue_pos_choices_live) >=1:
                self.state = 2
            elif len(self.EEG_cue_pos_choices_notlive) >= 1:
                return task.again
            elif len(self.EMG_cue_pos_choices_live) <=1 and len(self.EEG_cue_pos_choices_live) < 1:
                self.state = 0

            return task.again
        
        ### live EEG, same principle as live EMG
        elif self.state == 13 or self.state == 13.5:
            if  len(self.EEG_cue_pos_choices_live) <1 and len(self.EMG_cue_pos_choices_live) <=1:
                self.state = 0 
                return task.again
            elif len(self.EEG_cue_pos_choices_live) <1:
                self.state -= 10  
                return task.again
            else:
                self.EEG_cue_pos_choices_live = self.show_motor_imagery(self.EEG_cue_pos_choices_live)
                ### change to 4 or 4.5, important to keep information of high or low
            if  len(self.EMG_cue_pos_choices_live) <=1:
                self.state -= 9
            else:
                self.state -= 10
                   
            return task.again
        
        ### other, e.g. 0:   shutdown
        else:            
            self.stop_session()
            return task.done



###############################################################################
    ### shut everything down that has been started in order to have a smooth restart next time
    def stop_session(self):  
        if self.mode == "EEG" or self.mode == "HYBRID":
            if (self.EEG_true+self.EEG_false) != 0:
                print('EEG Correct:',self.EEG_true,'/',self.EEG_true+self.EEG_false)
            self.record_data_EEG.stop_recording_and_dump()
            pygame.quit()    
        if self.mode == "EMG" or self.mode == "HYBRID":
             if (self.EMG_true+self.EMG_false) != 0:
                 print('EMG Correct:',self.EMG_true,'/',self.EMG_true+self.EMG_false)
             self.record_data_EMG.stop_recording_and_dump()
       
            
        ShowBase.destroy(self)#base.destroy()
        if self.robothandconnected:
            self.robothand.shutdown()
        if self.robotarmconnected:
            self.robotarm.shutdown()
        sys.exit()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="emg experiment with panda visualisation")
    parser.add_argument("-t", "--trials"       , help="number of trials"        , default=72   , type=int)
    parser.add_argument("-f", "--Fs"           , help="sampling frequency"      , required=True, type=int)
    parser.add_argument("-a", "--age"          , help="age of the subject"      , required=True, type=int)
    parser.add_argument("-g", "--gender"       , help="gender of the subject"   , required=True)
    parser.add_argument("-w", "--with_feedback", help="with additional feedback", type=bool)
    parser.add_argument("-l", "--trials_notlive",help="number of trials before live", default=66,type=int)
    parser.add_argument("-p", "--force_classify",help="enable force classification", default=0, type=int)
    parser.add_argument("-m", "--mode"          ,help="EEG, EMG or HYBRID"      , required=True)

    args = vars(parser.parse_args())

    app = hybridBCI(args["force_classify"], args['trials'], args["trials_notlive"], args["mode"], args["Fs"], args["age"],
            gender=args["gender"], with_feedback=args["with_feedback"])
    app.run() 
    
    
    
"""
        def run_notlive_EEG(self):
        ### pick any random cue out of the list
        cue_pos = random.choice(list(self.EEG_cue_pos_choices_notlive.keys()))
        self.EEG_cue_pos_choices_notlive[cue_pos] -= 1
        if self.EEG_cue_pos_choices_notlive[cue_pos] == 0:
            del self.EEG_cue_pos_choices_notlive[cue_pos]
        print(self.EEG_cue_pos_choices_notlive)
        print(len(self.EEG_cue_pos_choices_notlive))
        ### perform the paradigm related to the cue
        self.cue_pos_choices_notlive = self.show_motor_imagery(self.cue_pos_choices_notlive)

    def run_live_EEG(self):   
        ### pick any random cue out of the list
        cue_pos = random.choice(list(self.EEG_cue_pos_choices_live.keys()))
        self.EEG_cue_pos_choices_live[cue_pos] -= 1
        if self.EEG_cue_pos_choices_live[cue_pos] == 0:
            del self.EEG_cue_pos_choices_live[cue_pos]            
        print(self.EEG_cue_pos_choices_live)
        print(len(self.EEG_cue_pos_choices_notlive))
        ### perform the paradigm related to the cue
        self.cue_pos_choices_notlive = self.show_motor_imagery(self.cue_pos_choices_live)
"""
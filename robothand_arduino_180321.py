# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
# 21.03.2018
# this class is used for communication with the robot hand with aid of serial ports
# in direct communication with the serial port, i.e. the write function:
#       0 = open hand, 1 = fist, 2 = pinch_2, 3 = pinch_3
# when interfacing the class with do_posture(), the same indices are used as in run_session_liveEMG.py
#       0 = fist, 1 = pinch_2, 2 = pinch_3 -->in do_posture() 1 is added to the input of the function

import serial
import time
import numpy as np

class RobotHand():
    def __init__(self, port="COM4",baudrate=38400):
        ### port is COM4 on recording PC and COM6 on Jonas' PC
        ### baudrate has to be the one defined in the Arduino Sketch

        self.serialPort = serial.Serial(port, baudrate) 
        
        ### make sure that it is newly opened
        if self.serialPort.isOpen():
            self.serialPort.close()
        self.serialPort.open()        
        
        ### initialise hand to open hand
        self.serialPort.write(bytes(chr(0),'UTF-8'))
        print(self.serialPort.readline())
        time.sleep(5)
        


###############################################################################        
    def do_posture(self, posture=0, duration=5):
        ### do posture for a duration and then relax for the same time
        ### make sure both high and low force postures give the same outcome
        posture = np.mod(posture,10)
        self.serialPort.write(bytes(chr(posture+1),'UTF-8'))
        #print(self.serialPort.readline())

        time.sleep(duration)
        self.serialPort.write(bytes(chr(0),'UTF-8'))
        #print(self.serialPort.readline())
        time.sleep(duration)



###############################################################################        
    def shutdown(self):
        self.serialPort.close()
        


###############################################################################        
if __name__ == '__main__':
    robothand = RobotHand("COM6")
    print('RobotHand is initialised. Now doing postures 0,1,2')
    time.sleep(5)
    robothand.do_posture(0)
    robothand.do_posture(1)
    robothand.do_posture(2)
    robothand.do_posture(6) ### secret posture
    robothand.shutdown()



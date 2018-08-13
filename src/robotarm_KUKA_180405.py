# Jonas Braun, jonas.braun@tum.de
# MSNE Research Internship Hybrid BCI
#05.04.2018
# this class is used for communication with the KUKA robot arm with aid of a UDP socket
# in direct communication with UDP
#       0 = move left, 1 = move right, 2 = come back, 3 = move down, 4= end session
# commands received in do_posture():
#       0 = move left, 1 = move right, 2 = both hands -> move down

#based on: https://pymotw.com/3/socket/udp.html

import time
import socket


class RobotArm():
    def __init__(self):
        # Create a UDP socket
        print('Creating UDP socket')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Define server address of control PC
        self.server_address = ('10.162.214.50', 8866)
        message = bytes(chr(int(2)),'UTF-8')
        self.sock.sendto(message, self.server_address)
        time.sleep(3)



    def do_posture(self,pos):
        if pos == 2:
            pos += 1
        message = bytes(chr(int(pos)),'UTF-8')
        self.sock.sendto(message, self.server_address)
        time.sleep(3)



    def return_home(self):
        message = bytes(chr(int(2)),'UTF-8')
        self.sock.sendto(message, self.server_address)
        time.sleep(3)



    def shutdown(self):
        self.return_home()
        message = bytes(chr(int(4)),'UTF-8')
        self.sock.sendto(message, self.server_address)
        print('Closing UDP socket')
        self.sock.close()



if __name__ == '__main__':
    myarm = RobotArm()

    myarm.do_posture(0)
    time.sleep(5)
    myarm.return_home()
    time.sleep(5)
    myarm.do_posture(1)
    time.sleep(5)
    myarm.return_home()
    time.sleep(5)
    myarm.do_posture(2)
    time.sleep(5)
    myarm.return_home()

    myarm.shutdown()

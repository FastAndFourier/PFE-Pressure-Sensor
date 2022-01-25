import serial
import time
import numpy as np

def servo_full_sweep(arduino,print_=False,sleep=0):

    #time.sleep(1)
    # Init each servo to 0
    arduino.write(('0/0\n').encode())  

    arduino.readline()

    for pos_pan in range(20,130,10):
        for pos_tilt in range(50,130,10):
            pos = str(pos_pan) + '/' + str(pos_tilt) + '\n'
            arduino.write(pos.encode())
            arduino.read(4).decode()
            #arduino.read(5).decode()
            #print(arduino.read(5).decode().split('\r\n')[0])
            #arduino.readline()

            # while measure!="received":
            #     measure = arduino.read(8).decode().split('\r\n')[0]

            time.sleep(.1)
            
            
            
    arduino.write(('0/0\n').encode()) 




if __name__ == "__main__":

    serial_port = serial.Serial(port="COM4", baudrate = 115200,timeout=0.9)

    servo_full_sweep(serial_port,print_=False,sleep=1)
    serial_port.close()
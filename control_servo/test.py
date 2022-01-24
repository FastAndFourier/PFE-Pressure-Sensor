import serial
import time
import numpy as np

def blink(arduino,frequency):
    
    time_start = time.time()
    total_time = 0

    while total_time < 10:
        arduino.writelines(b'H')
        time.sleep(1/(2*frequency))
        #print("in")
        arduino.writelines(b'L')
        total_time = time.time() - time_start
        

if __name__ == "__main__":

    serial_port = serial.Serial(port="COM3", baudrate = 9600,timeout=1)

    # # réinitialisation
    # serial_port.setDTR(False)
    # time.sleep(0.1)
    # serial_port.setDTR(True)

    # # on vide le buffer
    # serial_port.flushInput()

    serial_port.write(('0\n').encode())    

    for pos_pan in range(20,150,10):
        for pos_tilt in range(50,130,10):

            pos = str(pos_pan) + '/' + str(pos_tilt) + '\n'

            serial_port.write(pos.encode())
        
            read_data = serial_port.readlines()
            if len(read_data)!=0:
                print(read_data[0].decode())

    serial_port.close()
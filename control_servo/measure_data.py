import serial
import time

if __name__ == "__main__":

    serial_port = serial.Serial(port="COM4", baudrate = 115200,timeout=1)

    read_accel = False
    read_pressure = False

    for tps in range(1000):
        print(serial_port.readline().decode().split("\r\n")[0])
        if serial_port.readline().decode().split("\r\n")[0]=='a':
            read_accel = True
        
        while read_accel:
            time.sleep(.5)
            print(serial_port.readline().decode().split("\r\n")[0])


    serial_port.close()

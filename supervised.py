import serial
import keyboard
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def read_captor():
    serial_port = serial.Serial(port="COM5", baudrate = 9600, timeout = 1)
    # réinitialisation
    serial_port.setDTR(False)
    time.sleep(0.1)
    serial_port.setDTR(True)
    
    # on vide le buffer
    serial_port.flushInput()
    
    # lecture des données
    taillex = 18#nombre de cellules du capteur
    tailley = 9
    taille = taillex*tailley
    
    tablel = np.zeros((taille,1))
    #read_table = 0
    index = np.array([[i,j] for i in range(0,9,2) for j in range(0,5,2)])
    
    if (int(serial_port.readline())==1028):
        read_table=1
        
    j = 0
    
    while(read_table&(j<taille)):
        #print(serial_port.readline())
        tablel[j] = int(serial_port.readline())
        j+=1
    
    table=tablel.reshape((taillex,tailley))
    
    table_img = cv2.resize((table.T*255/1024).astype(np.uint8),(taillex*50,tailley*50))   
    heatmap = cv2.applyColorMap(table_img, cv2.COLORMAP_HOT)
    
    # cv2.imshow('Recording',heatmap)

    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    
    serial_port.close()
    return table
    
def read_keyboard():
    fst_dir = snd_dir = 0
    
    print("\x1b[K Right or Left ?", end=' ')
    while True:
        key = keyboard.read_hotkey()
        if keyboard.is_pressed("droite") or keyboard.is_pressed("right"):
            print("You pressed right", end = '\r')
            fst_dir = 1
            break
            
        elif keyboard.is_pressed("gauche") or keyboard.is_pressed("left"):
            print("You pressed left", end = '\r')
            fst_dir = -1
            break
        
        elif keyboard.is_pressed("space") or keyboard.is_pressed("enter"):
            print("No direction", end = '\r')
            break
        
            
    print("\x1b[K Up or Down ?", end = ' ')
    while True:
        key = keyboard.read_key()
        if keyboard.is_pressed("haut") or keyboard.is_pressed("up"):
            print("You pressed up",  end = '\r')
            snd_dir = 1
            break
        
        elif keyboard.is_pressed("bas") or keyboard.is_pressed("down"):
            print("You pressed down",  end = '\r')
            snd_dir = -1
            break
        
        elif keyboard.is_pressed("enter") or  keyboard.is_pressed("space"):
            print("No direction",  end = '\r')
            break
        
    return fst_dir, snd_dir


def servo_full_sweep(arduino, print_=False,sleep=0):

    #time.sleep(1)
    # Init each servo to 0
    arduino.write(('0/0\n').encode())  

    arduino.readline()
    plt.figure()
    #pan = bas, tilt = haut
    for pos_pan in [50]:
        for pos_tilt in [50]:
            #print(pos_pan)
            #print(pos_tilt)
            pos = str(50) + '/' + str(50) + '\n'
            #print(pos.encode())
            arduino.write(pos.encode())
            
            time.sleep(.1)
            #image_data = read_captor()
            
            #table_img = cv2.resize((image_data.T*255/1024).astype(np.uint8),(18*50,9*50))   
            #heatmap = cv2.applyColorMap(table_img, cv2.COLORMAP_HOT)

       

            #plt.figimage(heatmap)
            #cv2.waitKey(1)
            #cv2.destroyAllWindows() 
            #plt.draw()
            #plt.pause(0.001)
            #keyboard_input = read_keyboard()
            
            
       
    arduino.write(('5/5\n').encode()) 

if __name__ == "__main__":
    print("Recording Start!")
    #plt.ion()
    serial_port = serial.Serial(port="COM6", baudrate = 115200,timeout=0.9)
    
    #image_data = read_captor()
    servo_full_sweep(serial_port, print_=False,sleep=1)
    plt.ioff()
    #print(image_data)
    serial_port.close()
    

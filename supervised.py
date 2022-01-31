import serial
import keyboard
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

import torch as th

from control_policy import policyNet, predict

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


def get_expert_action(pos_pan,pos_tilt):

    pos_center_pan= 70
    pos_center_tilt = 80#90

    advice = []

    if(pos_pan>pos_center_pan):
        advice.append(-1)
    elif(pos_pan==pos_center_pan):
        advice.append(0)
    elif(pos_pan<pos_center_pan):
        advice.append(1)

    if(pos_tilt>pos_center_tilt):
        advice.append(-1)
    elif(pos_tilt==pos_center_tilt):
        advice.append(0)
    elif(pos_tilt<pos_center_tilt):
        advice.append(1)

    return advice

def states_to_actions(arduino, print_=False,sleep=0):

    #time.sleep(1)
    # Init each servo to 0
    #arduino.write(('0/0\n').encode())  

    #arduino.readline()
    plt.figure()
    #pan = bas, tilt = haut

    pos_pan = 0
    pos_tilt = 0

    demonstration = []

    incr = 10

    observation = []
    action = []

    # print("Zero measure")
    # measure = input()
    # while measure!='.':
    #     measure = input()
    
    # zero = read_captor()
    # print("zero done")
    # observation.append(zero)

    policy_network = th.load("./expert_policy.pt")

    print(type(policy_network))

    

    for nb_step in range(0,10,1):

        measure = input()
        while measure!='.':
            measure = input()
        
        obs = read_captor()
        observation.append(obs)
        print(obs)
        act = predict(np.array(obs),policy_network)#get_expert_action(pos_pan,pos_tilt)
        print(act)

        #action.append(act)
        #demonstration.append([obs,action])
        # measure = input()
        # while measure not in ['00','10','01','11']:
        #     measure = input()

        pos = str(act[0]) + '/' + str(act[1]) + '\n'
        #print(pos.encode())
        arduino.write(pos.encode())

        pos_tilt += incr

        #if(i_tilt == 130): print("Last one be ready")
            
        # np.save('./observation3.npy',np.array(observation),allow_pickle=True)
        # np.save('./action3.npy',np.array(action),allow_pickle=True)
        #print("return")
        #pos = str(10) + '/' + str(-170) + '\n'
        arduino.write(pos.encode())

        # pos_pan += incr
        # pos_tilt = 0

def servo_full_sweep(arduino, print_=False,sleep=0):

    #time.sleep(1)
    # Init each servo to 0
    #arduino.write(('0/0\n').encode())  

    #arduino.readline()
    plt.figure()
    #pan = bas, tilt = haut

    
    # pos_center = str(pos_center_pan) + '/' + str(pos_center_tilt) + '\n'

    # advice_pan = []
    # advice_tilt = []
    # #arduino.write(pos_center.encode())
    # # start_tilt = np.random.rand()
    # # start_pan =
    # val_plus = np.arange(start=pos_center_pan, stop = 130 ,step = 1)
    # val_moins = np.arange(start=pos_center_tilt, stop = 70 ,step = -1)
    # values = []
 
    # for i in range(len(val_plus)):
    #     values.append(val_plus[i])
    #     values.append(val_moins[i])
    # print(values)

    # for pos_pan in values:
    #     for pos_tilt in values:

    pos_pan = 0
    pos_tilt = 0

    demonstration = []

    incr = 10

    
    observation = []
    action = []

    print("Zero measure")
    measure = input()
    while measure!='.':
        measure = input()
    
    zero = read_captor()
    print("zero done")
    observation.append(zero)

    for i_pan in range(0,130,incr):
        for i_tilt in range(0,150,incr):
            print(f"{i_pan} {i_tilt}")
            measure = input()
            while measure!='.':
                measure = input()
            
            obs = read_captor()
            observation.append(obs)
            print(obs)
            act = get_expert_action(pos_pan,pos_tilt)
            print(act)
            action.append(act)
            #demonstration.append([obs,action])
            # measure = input()
            # while measure not in ['00','10','01','11']:
            #     measure = input()


            pos = str(0) + '/' + str(10) + '\n'
            #print(pos.encode())
            arduino.write(pos.encode())

            pos_tilt += incr

            if(i_tilt == 130): print("Last one be ready")
            
        np.save('./observation3.npy',np.array(observation),allow_pickle=True)
        np.save('./action3.npy',np.array(action),allow_pickle=True)
        print("return")
        pos = str(10) + '/' + str(-170) + '\n'
        arduino.write(pos.encode())

        pos_pan += incr
        pos_tilt = 0

    #np.save('./demo.npy',np.array(demonstration,dtype=list))
        
        # for pos_tilt2 in range(pos_tilt, 10, -10):
        #     print(pos_tilt2)
        #     pos = str(pos_pan) + '/' + str(pos_tilt2) + '\n'
        #     #print(pos.encode())
        #     arduino.write(pos.encode())
        #time.sleep(1)
        # if(pos_pan>pos_center_pan):
        #     advice_pan.append(-1)
        # elif(pos_pan==pos_center_pan):
        #     advice_pan.append(0)
        # elif(pos_pan<pos_center_pan):
        #     advice_pan.append(1)

        # if(pos_tilt>pos_center_tilt):
        #     advice_tilt.append(-1)
        # elif(pos_tilt==pos_center_tilt):
        #     advice_tilt.append(0)
        # elif(pos_tilt<pos_center_tilt):
        #     advice_tilt.append(1)

        #time.sleep(1)

            # image_data = read_captor()
            
            # table_img = cv2.resize((image_data.T*255/1024).astype(np.uint8),(18*50,9*50))   
            # heatmap = cv2.applyColorMap(table_img, cv2.COLORMAP_HOT)

       

            # plt.figimage(heatmap)
            # #cv2.waitKey(1)
            # #cv2.destroyAllWindows() 
            # plt.draw()
            # plt.pause(0.5)
            #keyboard_input = read_keyboard()
            
            
            
    # print("pan = ",advice_pan)
    # print("tilt=", advice_tilt)
       
    #arduino.write(('0/0\n').encode()) 

if __name__ == "__main__":
    print("Recording Start!")
    #plt.ion()
    serial_port = serial.Serial(port="COM6", baudrate = 115200,timeout=0.9)
    # print(serial_port.readline())
    # #image_data = read_captor()
    states_to_actions(serial_port, print_=False,sleep=1)
    # plt.ioff()
    # #print(image_data)
    serial_port.close()
    

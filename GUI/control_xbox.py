import pygame
import serial
import time
import numpy as np

import cv2


# Define some colors.
BLACK = pygame.Color('black')
WHITE = pygame.Color('white')


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

class Servo_motors:
    def __init__(self):
        self.min=0
        self.max=180
        self.pos1 = 0
        self.pos2 = 0
        self.margin = 20

    def reset_s1(self):
        self.pos1=self.min

    def reset_s2(self):
        self.pos2=self.min

    def reset(self):
        self.reset_s1()
        self.reset_s2()

    def forward(self, arduino, msg):
        if(msg=="20"):
            if self.pos1 <= self.max-self.margin:
                pos = 1 #self.pos1 + 1
                new_pos = "0" + '/' + str(pos) + '\n'
                arduino.write(new_pos.encode())
                self.pos1+=1 #=pos
            return self.pos1
        elif(msg=="02"):
            if self.pos2 <= self.max-self.margin:
                pos = 1 #self.pos2 + 1
                new_pos = str(pos) + '/' +  "0" + '\n' #str(self.pos1)
                arduino.write(new_pos.encode())
                self.pos2+=1 #=pos
            return self.pos2
        else:
            return None

    def backward(self,arduino, msg):
        if(msg=="10"):
            if self.pos1 >= self.min+self.margin:
                pos = - 1 #self.pos1 - 1
                new_pos = "0" + '/' + str(pos) + '\n'
                arduino.write(new_pos.encode())
                self.pos1-=1#=pos
            return self.pos1
        elif(msg=="01"):
            if self.pos2 >= self.min+self.margin:
                pos = - 1 #self.pos2 - 1
                new_pos = str(pos) + '/' + "0" + '\n'
                arduino.write(new_pos.encode())
                self.pos2-=1#=pos
            return self.pos2
        else:
            return None
    

if __name__ == "__main__":
    pygame.init()

    pygame.display.set_caption("Motor Control")

    done = False

    clock = pygame.time.Clock()

    pygame.joystick.init()

    #Instantiate the Servos for the leg
    s1=Servo_motors()

    #Connect to Serial port
    arduino = serial.Serial(port="COM6", baudrate = 115200, timeout=0.01)
    old_pos = None
    old_event = None
    
    mode = 0

    if(pygame.joystick.get_count()>0):
        print("Joystick detected, joystick mode")
        mode = 1
        joystick_count = pygame.joystick.get_count()
        # For each joystick:
        for i in range(joystick_count):
            joystick = pygame.joystick.Joystick(i)
            joystick.init()

            try:
                jid = joystick.get_instance_id()
            except AttributeError:
                # get_instance_id() is an SDL2 method
                jid = joystick.get_id()
    else:
        print("Joystick not detected, keyboard mode")
        mode = 0

    end = False

    while not done:

    #Code for msg is 2:forward/1:backward/0:stop
    #And there are 2 motors, so the msg is of size 2 : example : "20"
        events = pygame.event.get()

        if old_event is not None and events == [] and end==False:
            events = old_event

        for event in events: # User did something.
            end=False
            if mode==0:
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                    #print("Button 0 used.")
                    old_pos = s1.forward(arduino, "20")
                    old_event = events
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                    #print("Button 1 used.")
                    old_pos = s1.backward(arduino, msg = "10")
                    old_event = events
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                    #print("Button 2 used.")
                    old_pos = s1.forward(arduino, "02")
                    old_event = events
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                    #print("Button 4 used.")
                    old_pos = s1.backward(arduino, msg = "01")
                    old_event = events 
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    table = read_captor()
                    print(table) 
                    old_event = events  
                elif event.type == pygame.KEYUP and event.key in [pygame.K_RIGHT,pygame.K_LEFT,pygame.K_DOWN,pygame.K_UP, pygame.K_SPACE] and end == False:# and event.key == pygame.K_LEFT:  
                    old_event = events
                    end = True  

            elif mode == 1:
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.JOYBUTTONDOWN and event.button == 3:
                    #print("Button 0 used.")
                    old_pos = s1.forward(arduino, "20")
                    old_event = events
                elif event.type == pygame.JOYBUTTONDOWN and event.button == 0:
                    #print("Button 1 used.")
                    old_pos = s1.backward(arduino, msg = "10")
                    old_event = events
                elif event.type == pygame.JOYBUTTONDOWN and event.button == 1:
                    #print("Button 2 used.")
                    old_pos = s1.forward(arduino, "02")
                    old_event = events
                elif event.type == pygame.JOYBUTTONDOWN and event.button == 2:
                    #print("Button 4 used.")
                    old_pos = s1.backward(arduino, msg = "01")
                    old_event = events
                elif event.type == pygame.JOYBUTTONDOWN and event.button == 5:
                    table = read_captor()
                    print(table)

                elif event.type == pygame.JOYBUTTONUP and event.button in [0,1,2,3,5] and end==False:     
                    old_event = events
                    end=True

        # pygame.display.flip()

        clock.tick(100)

    pygame.quit()
import pygame
import serial
import time
import numpy as np


# Define some colors.
BLACK = pygame.Color('black')
WHITE = pygame.Color('white')

class Servo_motors:
    def __init__(self):
        self.min=0
        self.max=180
        self.pos1 = 0
        self.pos2 = 0
        self.margin = 0#20

    def reset_s1(self):
        self.pos1=self.min

    def reset_s2(self):
        self.pos2=self.min

    def reset(self):
        self.reset_s1()
        self.reset_s2()

    def forward(self, arduino, msg):
        arduino.readline()
        if(msg=="20"):
            if self.pos1 <= self.max-self.margin:
                pos = self.pos1 + 2
                new_pos = str(self.pos2) + '/' + str(pos) + '\n'
                arduino.write(new_pos.encode())
                arduino.read(4).decode()
                self.pos1=pos
            return self.pos1
        elif(msg=="02"):
            if self.pos2 <= self.max-self.margin:
                pos = self.pos2 + 2
                new_pos = str(pos) + '/' + str(self.pos1) + '\n'
                arduino.write(new_pos.encode())
                arduino.read(4).decode()
                self.pos2=pos
            return self.pos2
        else:
            return None

    def backward(self,arduino, msg):
        arduino.readline()
        if(msg=="10"):
            if self.pos1 >= self.min+self.margin:
                pos = self.pos1 - 2
                new_pos = str(self.pos2) + '/' + str(pos) + '\n'
                arduino.write(new_pos.encode())
                arduino.read(4).decode()
                self.pos1=pos
            return self.pos1
        elif(msg=="01"):
            if self.pos2 >= self.min+self.margin:
                pos = self.pos2 - 2
                new_pos = str(pos) + '/' + str(self.pos1) + '\n'
                arduino.write(new_pos.encode())
                arduino.read(4).decode()
                self.pos2=pos
            return self.pos2
        else:
            return None

def control_4_motors(arduino, print_=False, sleep=0, msg = "0000"):

    #time.sleep(1)
    # Init each servo to 0
    # arduino.write(('0/0\n').encode())  
    # arduino.readline()

    pos = str(3) + '/' + str(msg) + '\n'
    arduino.write(pos.encode())
    arduino.read(4).decode()
    time.sleep(.1)  
    arduino.write(('/0\n').encode()) 

def control_2_motors(arduino, old_pos = None,print_=False,sleep=0, msg = "00"):

    #time.sleep(1)
    # Init each servo to 0
    #arduino.write(('0/0\n').encode())  
    print("MOVE")
    arduino.readline()

    if msg == "00":
        return old_pos

    if old_pos is None:
        pos = 5
    elif msg == "20":
        pos = old_pos + 2
    elif msg == "10":
        pos = old_pos - 2

    if pos < 130:
    # for pos_pan in range(20,130,10):
    #     for pos_tilt in range(50,130,10):
        new_pos = str(15) + '/' + str(pos) + '\n'
        arduino.write(new_pos.encode())
        arduino.read(4).decode()
    return pos
       
class TextPrint(object):
    def __init__(self):
        self.reset()
        self.font = pygame.font.Font(None, 20)

    def tprint(self, screen, textString):
        textBitmap = self.font.render(textString, True, BLACK)
        screen.blit(textBitmap, (self.x, self.y))
        self.y += self.line_height

    def reset(self):
        self.x = 10
        self.y = 10
        self.line_height = 15

    def indent(self):
        self.x += 10

    def unindent(self):
        self.x -= 10

class Menu(TextPrint):
    def __init__(self):
        super().__init__()
        self.font = pygame.font.SysFont('Corbel', 20)
        self.color = (0,0,0)
        self.color_back = (255,10,20)
        self.buttons = []

    def create_window(self, width, height):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))

    def add_button(self, text, pos_x, pos_y, margins):
        text = self.font.render(text , True , self.color)
        self.buttons.append([pos_x/2, pos_y/2])
        self.margins = margins
        pygame.draw.rect(self.screen, self.color_back, [pos_x/2, pos_y/2, margins[0], margins[1]])
        self.screen.blit(text , (pos_x/2+50, pos_y/2))

    def click_button(self, mouse):
        # print(mouse)
        # print(self.buttons)
        
        if ((self.buttons[0][0])-self.margins[1]+30) <= mouse[0] <= ((self.buttons[0][0])+self.margins[1]+30) and ((self.buttons[0][0])-self.margins[0]+10) <= mouse[1] <= (self.buttons[0][1])+(self.margins[0]/2+10):
            print("Clicked on Leg")

        elif ((self.buttons[1][0])-self.margins[1]+30) <= mouse[0] <= ((self.buttons[1][0])+self.margins[1]+30) or ((self.buttons[1][1])-self.margins[0]+50) <= mouse[1] <= (self.buttons[1][1])+(self.margins[0])/2+50:
            print("Clicked on Arm")

class Motor_control(Menu):
    def __init__(self):
        super.__init__()
    
    def send_command(self):
        fond = pygame.image.load("../Xbox_controller.png").convert()
        self.screen.blit(fond, (0,0))

pygame.init()

pygame.display.set_caption("Motor Control")

done = False

clock = pygame.time.Clock()

pygame.joystick.init()

textPrint = TextPrint()
menu = Menu()

menu.reset()
menu.create_window(200, 200)
menu.screen.fill(WHITE)
menu.add_button("leg", 60, 40, [140, 40])
menu.add_button("arm", 60, 200, [140, 40])
cmd = "0000"

#Instantiate the Servos for the leg
s1=Servo_motors()

#Connect to Serial port
arduino = serial.Serial(port="COM6", baudrate = 115200, timeout=0.01)
old_pos = None
old_event = None

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

while not done:

#Code for msg is 2:forward/1:backward/0:stop
#And there are 2 motors, so the msg is of size 2 : example : "20"

    events = pygame.event.get()

    if old_event is not None and events == []:
        events = old_event

    for event in events: # User did something.

        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.JOYBUTTONDOWN and event.button == 3:
            #print("Button 0 used.")
            #cmd = "2000"
            old_pos = s1.forward(arduino, "20")
            old_event = events
        elif event.type == pygame.JOYBUTTONDOWN and event.button == 0:
            #print("Button 1 used.")
            #cmd = "0200"
            old_pos = s1.backward(arduino, msg = "10")
            old_event = events

        elif event.type == pygame.JOYBUTTONDOWN and event.button == 1:
            #print("Button 2 used.")
            #cmd = "0020"
            old_pos = s1.forward(arduino, "02")
            old_event = events

        elif event.type == pygame.JOYBUTTONDOWN and event.button == 2:
            #print("Button 4 used.")
            #cmd = "0002"
            old_pos = s1.backward(arduino, msg = "01")
            old_event = events

        elif event.type == pygame.JOYBUTTONUP and (event.button == 0 or event.button == 1 or event.button == 2 or event.button == 3):
            #print("stop")
            #cmd = "0000"
            old_event = events
        

    pygame.display.flip()

    clock.tick(100)

pygame.quit()
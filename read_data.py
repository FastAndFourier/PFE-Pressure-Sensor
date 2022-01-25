import serial
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
#import tensorflow as tf

if __name__ == "__main__":

    serial_port = serial.Serial(port="COM5", baudrate = 9600)

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
    temps_record = 2000
    tablel = np.zeros((taille,1))
    table_through_time = np.zeros((taillex,tailley,temps_record))

    index = np.array([[i,j] for i in range(0,9,2) for j in range(0,5,2)])


    record_data = []
    time_ = time.time()

    for i in range(temps_record):
        read_table=0
        j=0
        if (int(serial_port.readline())==1028):
            read_table=1
        while(read_table&(j<taille)):
            #print(serial_port.readline())
            tablel[j] = int(serial_port.readline())
            j+=1


        table=tablel.reshape((taillex,tailley))
        table_through_time[:,:,i] = table 
        
        table_img = cv2.resize((table.T*255/1024).astype(np.uint8),(taillex*50,tailley*50))   
        heatmap = cv2.applyColorMap(table_img, cv2.COLORMAP_HOT)

        print(time.time()-time_)
        time_ = time.time()

        cv2.imshow('Recording',heatmap)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        

    serial_port.close()
    cv2.destroyAllWindows()

    np.save("table_through_time_2", table_through_time)
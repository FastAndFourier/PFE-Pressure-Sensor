### Author : Louis Simon ###
## Sensor cells selection for pressure maxima estimation ##

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from PIL import Image
import os
from tqdm import tqdm

from build_NN import *



def process_data(I,index,offset):

    data = []
    for k in range(I.shape[0]):
        if k < offset:
            data.append(I[k,index[:,0],index[:,1]])
        else:
            #data.append((I[k,index[:,0],index[:,1]]-I[k-offset,index[:,0],index[:,1]]))
            data.append(np.mean(I[k-offset:k+1,index[:,0],index[:,1]],axis=0))

    
    return np.array(data)

def get_label(X,size,unravel=True):
    data = []
    if unravel:
        for x in X:
            data.append(np.array(np.unravel_index(np.argmin(x),size),dtype=float))
    else:
        for x in X:
            data.append(np.argmin(x)*1.0)

    return np.array(data)



if __name__ == "__main__":

    data = np.transpose(np.load("table_through_time_2.npy"),(2,0,1))
    sizex, sizey = data.shape[1], data.shape[2]


    # Full index
    # index = np.array([[i,j] for i in range(9) for j in range(5)])

    # Sampling on columns
    # index = np.array([[0,0],[0,2],[0,4],
    #                   [1,1],[1,3],
    #                   [2,0],[2,2],[2,4],
    #                   [3,1],[3,3],
    #                   [4,0],[4,2],[4,4],
    #                   [5,1],[5,3],
    #                   [6,0],[6,2],[6,4],
    #                   [7,1],[7,3],
    #                   [8,0],[8,2],[8,4]])

    # Sampling on columns + rows
    index = np.array([[i,j] for i in range(0,9,2) for j in range(0,5,2)])

    # Temporal offset for filtering
    offset = 2

    # Data processing
    X = process_data(data,index,offset)
    y = get_label(data,[data.shape[1],data.shape[2]],unravel=True)

    print(X.shape)

    
    lin_idx = np.array([i[0]*sizex + i[1] for i in y])
    print("Activated cells ",np.unique(lin_idx).size)
    dim = index.shape[0]
    
    # NN training
    X_train, X_test, y_train, y_test, scaler = process_data_nn(X,y,sizex,sizey)
    model = build_nn(dim,lr=0.005)

    callback = tf.keras.callbacks.ModelCheckpoint('./log/model1', save_best_only=True, monitor='accuracy', mode='max')

    model.summary()
    model.fit(x=X_train,y=y_train,
              epochs=100,batch_size=32,
              callbacks=[callback])

    score = model.evaluate(X_test,y_test)
    print("Accuracy = {:.2f} %".format(score[1]*100))



    # Creates a gif of prediction
    imgs = []

    print("Creating a gif...")
    for k in tqdm(range(offset,100)):
        idx = k
        data_display = (np.expand_dims(process_data(data[idx-offset:idx],index,offset)[-1],axis=0)-scaler[0])/scaler[1]
        y_display = np.array(model.predict(data_display)[0]*[sizex,sizey])
        y_display = np.round(y_display)
        y_gt = get_label([data[idx]],[data[idx].shape[0],data[idx].shape[1]],unravel=True)[0]

        figure = plt.figure()
        ax = figure.gca()

        ax.imshow(data[idx])
        ax.plot(y_display[1],y_display[0],'r+')
        ax.plot(y_gt[1],y_gt[0],'g*')

        ax.set_xticks([])
        ax.set_yticks([])

        width, height = figure.get_size_inches() * figure.get_dpi()
        canvas = FigureCanvas(figure)
        canvas.draw()
        

        im = np.frombuffer(canvas.tostring_rgb(),dtype=np.uint8).reshape(int(height), int(width), 3) 
        PIL_image = Image.fromarray(im).convert('RGB')

        plt.close()

        imgs.append(PIL_image)

    

    if not(os.path.isdir('img')):
        os.mkdir('img')

    imgs[0].save('./img/out_1.gif', save_all=True, optimize=False,append_images=imgs[0:],duration=200, loop=0)
   
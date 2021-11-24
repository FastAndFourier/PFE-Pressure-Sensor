import PIL
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import time
from scipy.signal import convolve2d
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR

import sklearn.metrics as metrics

from build_NN import *
from gif import *

from PIL import Image
import glob

import os

from tqdm import tqdm

def process_data(I,index,offset):

    data = []
    for k in range(I.shape[0]):
        if k < offset:
            data.append(I[k,index[:,0],index[:,1]])
        else:
            #data.append((I[k,index[:,0],index[:,1]]-I[k-offset,index[:,0],index[:,1]]))
            data.append(np.mean(I[k-offset:k+1,index[:,0],index[:,1]],axis=0))

    
    return np.array(data)

def get_label(X,unravel=True):
    data = []
    if unravel:
        for x in X:
            data.append(np.unravel_index(np.argmin(x),[9,5]))
    else:
        for x in X:
            data.append(np.argmin(x))

    return np.array(data)

        


if __name__ == "__main__":

    data = np.transpose(np.load("table_through_time.npy"),(2,0,1))
    sizex, sizey = data.shape[1], data.shape[2]

    # index = np.array([[i,j] for i in range(9) for j in range(5)])
    # index = np.array([[0,0],[0,2],[0,4],
    #                   [1,1],[1,3],
    #                   [2,0],[2,2],[2,4],
    #                   [3,1],[3,3],
    #                   [4,0],[4,2],[4,4],
    #                   [5,1],[5,3],
    #                   [6,0],[6,2],[6,4],
    #                   [7,1],[7,3],
    #                   [8,0],[8,2],[8,4]])
    index = np.array([[i,j] for i in range(0,9,2) for j in range(0,5,2)])

    offset = 3

    #index = np.array([[0,0],[sizex//2,0],[0,sizey//2],[sizex//2,sizey//2]])
    X = process_data(data,index,offset)
    print(X.shape)
    y = get_label(data,unravel=True)
    lin_idx = np.array([i[0]*sizex + i[1] for i in y])
    print(np.unique(lin_idx).size)
    dim = index.shape[0]
    
    X_train, X_test, y_train, y_test, scaler = process_data_nn(X,y,sizex,sizey)
    model = build_nn(dim,lr=0.005,dropout_rate=0.1)

    model.summary()
    model.fit(x=X_train,y=y_train,epochs=50,batch_size=32)

    score = model.evaluate(X_test,y_test)
    print("Accuracy = {:.2f} %".format(score[1]*100))

    plt.figure()

    #y_pred = model.predict(X_test)


    imgs = []

    for k in tqdm(range(offset,200)):
        idx = k#np.random.randint(offset,X_test.shape[0])
        data_display = (np.expand_dims(process_data(data[idx-offset:idx],index,offset)[-1],axis=0)-scaler[0])/scaler[1]
        y_display = np.array(model.predict(data_display)[0]*[sizex,sizey])
        y_gt = get_label([data[idx]],unravel=True)[0]

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
        
        #im, shape = canvas.print_to_buffer()

        #

        im = np.frombuffer(canvas.tostring_rgb(),dtype=np.uint8).reshape(int(height), int(width), 3) 
        PIL_image = Image.fromarray(im).convert('RGB')

        plt.close()
 
        # # Get the RGBA buffer from the figure
        # w,h = figure.canvas.get_width_height()
        # buf = np.frombuffer(figure.canvas.tostring_rgb(),dtype=np.uint8)
        # print(w*h*3)
        # buf.shape = (w,h,4)
    
        # # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        # buf = np.roll ( buf, 3, axis = 2 )

        # w, h, d = buf.shape
        # im = Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )

        

        # # fig.canvas.draw()

        # # im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        
        # # w, h = fig.canvas.get_width_height()
        
        # # im = im.reshape((int(h), int(w), 3))

        

        # plt.imshow(im)
        # plt.show()

        imgs.append(PIL_image)

        #plt.close()
        #os.remove(fname)
    

    if not(os.path.isdir('img')):
        os.mkdir('img')

    imgs[0].save('./img/out.gif', save_all=True, optimize=False,append_images=imgs[0:],duration=200, loop=0)
    # imgs = []
    # filenames = glob.glob('./img/img*.png')
    # filenames = sorted(filenames)
    # for f in filenames:
    #     imgs.append(Image.open(f))

    
    # imgs[0].save('./img/out.gif', save_all=True, optimize=False,append_images=imgs[0:],duration=200, loop=0)

    # model = RandomForestClassifier(n_estimators=30)
    # model.fit(X_train,y_train)

    # y_pred = model.predict(X_test)
    # print(metrics.accuracy_score(y_test,y_pred))
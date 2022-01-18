import numpy as np

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Dense, MaxPool1D, Dropout
from tensorflow.keras import Input, Model

def process_data_nn(X,y,sizex,sizey):
    # Splits the dataset and apply normalization to observations and labels

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    #X_train, X_test = (X_train-X.mean())/X.std(), (X_test-X.mean())/X.std()

    y_train = np.array([[l[0]/sizex,l[1]/sizey] for l in y_train])
    y_test = np.array([[l[0]/sizex,l[1]/sizey] for l in y_test])

    return X_train, X_test, y_train, y_test, [X.mean(),X.std()]

def build_nn(dim,lr=0.001):

    in_ = Input(shape=(dim,))
    
    x = Dense(units=dim,activation='relu')(in_)
    x = Dropout(rate=0.2)(x)
    x = Dense(units=dim//2,activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    # x = Dense(units=dim//4,activation='relu')(x)
    # x = Dropout(rate=dropout_rate)(x)
    output = Dense(units=2,activation='sigmoid')(x)
    #x = tf.keras.layers.Lambda(lambda x : tf.keras.backend.round(x)*1.0)(x)
    model = Model(in_,output)
    optimizer = tf.keras.optimizers.Adam(lr)
    model.compile(optimizer=optimizer,loss='mean_squared_error',metrics="accuracy")

    return model

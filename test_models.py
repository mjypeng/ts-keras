import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.layers import Input, Embedding, SimpleRNN, LSTM, Dense, merge, Lambda
from keras.models import Model, Sequential
from keras.utils.visualize_util import plot

# def basediff(x):
#     d  = x[:,:-1] - x[:,1:]
#     return K.concatenate([x[:,[0]],d],axis=1)
# outputs = Lambda(basediff,output_shape=lambda x: x)(inputs) # Output shape is the same as input shape

P  = 10
BaseDiff    = np.eye(P) + np.diag([-1.0]*(P-1), 1)
BaseDiffInv = np.triu(np.ones((P,P)))
inputs  = Input(shape=(P,))
x       = Dense(P,activation='relu',weights=[BaseDiff,np.zeros(P)])(inputs)
outputs = Dense(P,activation=None,weights=[BaseDiffInv,np.zeros(P)])(x)
MonotoneConstraint = Model(input=inputs,output=outputs)
MonotoneConstraint.layers[1].trainable  = False
MonotoneConstraint.layers[2].trainable  = False

x  = np.array([10,11,12,11,13,14,15,9,15,16])[None,:] #np.arange(10,0,-1)[None,:]
y  = model.predict(x)

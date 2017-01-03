import sys, os
sys.path.append('..' + os.sep + 'amazon' + os.sep + 'fba_mkd' + os.sep + 'ssm-python')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ssm

from keras.layers import Input, Embedding, LSTM, Dense, merge
from keras.models import Model, Sequential
from keras.utils.visualize_util import plot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#-- Read data --#
data = pd.read_csv('international-airline-passengers.csv',engine='python',skipfooter=3)
data.columns = ['month','passengers']
data['month'] = pd.to_datetime(data.month)
data.set_index('month',inplace=True)

n       = len(data)
time    = data.index
y       = data.passengers[None,:].astype('float32')
tr_size = int(len(data) * 0.67)
y_tr    = y[:,:tr_size]

#-- Analyze with Basic Structural Time Series Model --#
bstsm  = ssm.model_stsm('trend', 'trig1', 12)
bstsm  = ssm.estimate(y_tr,bstsm,np.log([10,1,1,0.1])/2,method='Nelder-Mead')[0]
alphahat,V = ssm.statesmo(np.hstack([y_tr,[[np.nan]*(n-tr_size)]]),bstsm)[:2]
y_hat  = ssm.signal(alphahat,bstsm,mcom='all')
RMSE_tr = np.sqrt(np.mean((y[:,:tr_size].squeeze()-y_hat[:tr_size])**2))
RMSE_tt = np.sqrt(np.mean((y[:,tr_size:].squeeze()-y_hat[tr_size:])**2))

# plt.plot(time,y.squeeze(),'b:o')
# plt.plot(time[:tr_size],y_hat[:tr_size],'g-^')
# plt.plot(time[tr_size:],y_hat[tr_size:],'r-^')
# plt.title("Training RMSE = %.2f,  Test RMSE = %.2f" % (RMSE_tr,RMSE_tt))
# plt.show()

#-- Analyze with RNN --#

# normalize the dataset
scaler  = MinMaxScaler(feature_range=(0,1))
scaler.fit(data.passengers[:tr_size])
data['scaled'] = scaler.transform(data.passengers)
for lag in range(1,25):
    data["lag%d" % lag] = np.hstack([np.nan]*lag + [data.scaled[:-lag].values])

def predict_nstep(model,X):
    n       = X.shape[0]
    max_lag = X.shape[2]
    y_hat   = np.vstack([X[0,:,:].T,[[np.nan]]*n])
    for t in range(max_lag,max_lag+n):
        y_hat[t] = model.predict(y_hat[t-max_lag:t].T[None,:,:])
    return y_hat[max_lag:]

trainPredict = {}
testPredict = {}
testPredict_nstep = {}
trainScore  = np.zeros(24)
testScore   = np.zeros(24)
testScore_nstep  = np.zeros(24)
for max_lag in range(1,25):
    col_lag = ["lag%d" % lag for lag in range(1,max_lag+1)]
    trainX  = data[max_lag:tr_size][col_lag].values[:,None,:]
    trainY  = data[max_lag:tr_size].scaled
    testX   = data[tr_size:][col_lag].values[:,None,:]
    testY   = data[tr_size:].scaled
    #
    np.random.seed(0) # fix random seed for reproducibility
    model  = Sequential()
    model.add(LSTM(4, input_length=1, input_dim=max_lag))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    res  = model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
    #
    trainPredict[max_lag] = scaler.inverse_transform(model.predict(trainX))
    testPredict[max_lag]  = scaler.inverse_transform(model.predict(testX))
    testPredict_nstep[max_lag]  = scaler.inverse_transform(predict_nstep(model,testX))
    #
    trainScore[max_lag-1] = np.sqrt(np.mean((data[max_lag:tr_size].passengers-trainPredict[max_lag].squeeze())**2))
    testScore[max_lag-1]  = np.sqrt(np.mean((data[tr_size:].passengers-testPredict[max_lag].squeeze())**2))
    testScore_nstep[max_lag-1]  = np.sqrt(np.mean((data[tr_size:].passengers-testPredict_nstep[max_lag].squeeze())**2))

plot(model, show_shapes=True, to_file='LSTM1.png')

def predict_nstep2(model,X):
    n       = X.shape[0]
    max_lag = X.shape[1]
    y_hat   = np.vstack([X[0,:,:],[[np.nan]]*n])
    for t in range(max_lag,max_lag+n):
        y_hat[t] = model.predict(y_hat[t-max_lag:t][None,:,:])
    return y_hat[max_lag:]

trainPredict2 = {}
testPredict2 = {}
testPredict_nstep2 = {}
trainScore2  = np.zeros(24)
testScore2   = np.zeros(24)
testScore_nstep2   = np.zeros(24)
for max_lag in range(1,25):
    col_lag = ["lag%d" % lag for lag in range(1,max_lag+1)]
    #
    trainX  = data[max_lag:tr_size][col_lag].values[:,:,None]
    trainY  = data[max_lag:tr_size].scaled
    testX   = data[tr_size:][col_lag].values[:,:,None]
    testY   = data[tr_size:].scaled
    #
    np.random.seed(0) # fix random seed for reproducibility
    model  = Sequential()
    model.add(LSTM(4, input_length=max_lag, input_dim=1))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    res  = model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
    #
    trainPredict2[max_lag] = scaler.inverse_transform(model.predict(trainX))
    testPredict2[max_lag]  = scaler.inverse_transform(model.predict(testX))
    testPredict_nstep2[max_lag]  = scaler.inverse_transform(predict_nstep2(model,testX))
    #
    trainScore2[max_lag-1] = np.sqrt(np.mean((data[max_lag:tr_size].passengers-trainPredict2[max_lag].squeeze())**2))
    testScore2[max_lag-1]  = np.sqrt(np.mean((data[tr_size:].passengers-testPredict2[max_lag].squeeze())**2))
    testScore_nstep2[max_lag-1]  = np.sqrt(np.mean((data[tr_size:].passengers-testPredict_nstep2[max_lag].squeeze())**2))

plot(model, show_shapes=True, to_file='LSTM2.png')

fig  = plt.figure(figsize=(16,10))
plt.plot(time,y.squeeze(),'b:o',label='airline')
plt.plot(time[tr_size:],y_hat[tr_size:],'g-^',label="BSTSM RMSE = %.2f" % RMSE_tt)
plt.plot(time[tr_size:],testPredict[12],'r:d',label="LSTM 1 RMSE = %.2f" % testScore[11])
plt.plot(time[tr_size:],testPredict2[12],'r-^',label="LSTM 2 RMSE = %.2f" % testScore2[11])
plt.legend()
plt.show()

fig  = plt.figure(figsize=(16,10))
plt.plot(time,y.squeeze(),'b:o',label='airline')
plt.plot(time[tr_size:],y_hat[tr_size:],'g-^',label="BSTSM RMSE = %.2f" % RMSE_tt)
plt.plot(time[tr_size:],testPredict_nstep[12],'r:d',label="LSTM 1 RMSE = %.2f" % testScore_nstep[11])
plt.plot(time[tr_size:],testPredict_nstep2[12],'r-^',label="LSTM 2 RMSE = %.2f" % testScore_nstep2[11])
plt.legend()
plt.show()

import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X13_PATH  = '..' + os.sep + 'winx13' + os.sep + 'x13as'
SSM_PATH  = '..' + os.sep + 'amazon' + os.sep + 'fba_mkd' + os.sep + 'ssm-python' #

sys.path.append(SSM_PATH)
import ssm

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.x13 import x13_arima_analysis

from keras.layers import Input, Embedding, SimpleRNN, LSTM, Dense, merge
from keras.models import Model, Sequential
from keras.utils.visualize_util import plot

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

MODEL_NAME  = sys.argv[1] #'MLP' #'SimpleRNN2' #'LSTM2' #'SimpleRNN1' #'LSTM1' #
MAX_LAG_RANGE = range(8,25)

#-- Read data --#
data = pd.read_csv('international-airline-passengers.csv',engine='python',skipfooter=3)
data.columns = ['month','passengers']
data['month'] = pd.to_datetime(data.month)
data.set_index('month',inplace=True)

n       = len(data)
time    = data.index
y       = data.passengers[None,:].astype('float32')
tr_size = int(np.floor(len(data) * 0.67))
y_tr    = y[:,:tr_size]

#-- Analyze with Basic Structural Time Series Model --#
bstsm  = ssm.model_stsm('trend', 'trig1', 12)
bstsm  = ssm.estimate(y_tr,bstsm,np.log([10,1,1,0.1])/2,method='Nelder-Mead')[0]
alphahat,V = ssm.statesmo(np.hstack([y_tr,[[np.nan]*(n-tr_size)]]),bstsm)[:2]
y_hat  = ssm.signal(alphahat,bstsm,mcom='all')
RMSE_tr = np.sqrt(np.mean((y[:,:tr_size].squeeze()-y_hat[:tr_size])**2))
RMSE_tt = np.sqrt(np.mean((y[:,tr_size:].squeeze()-y_hat[tr_size:])**2))
a,P    = ssm.kalman(y,bstsm)[:2]
y_hat_1step = ssm.signal(a,bstsm,mcom='all')
RMSE_tt_1step = np.sqrt(np.mean((y[:,tr_size:].squeeze()-y_hat_1step[tr_size:-1])**2))

# plt.plot(time,y.squeeze(),'b:o')
# plt.plot(time[:tr_size],y_hat[:tr_size],'g-^')
# plt.plot(time[tr_size:],y_hat[tr_size:],'r-^')
# plt.title("Training RMSE = %.2f,  Test RMSE = %.2f" % (RMSE_tr,RMSE_tt))
# plt.show()

# res  = x13_arima_analysis(data[:tr_size].passengers.astype(float),maxorder=(2,1),maxdiff=(2,1),trading=True,forecast_years=12,retspec=True,x12path=X13_PATH + os.sep + 'x13as.exe',prefer_x13=True)
# plt.plot(res.observed,'r:')
# plt.plot(res.seasadj,'b-')
# plt.plot(res.trend,'b--')
# plt.show()

# def objfunc(order, y):
#     res  = ARIMA(y, order).fit()
#     return res.aic()

#-- Analyze with NN --#

# normalize the dataset
scaler  = MinMaxScaler(feature_range=(0,1))
scaler.fit((data.passengers[:tr_size].astype(float))) #np.log
data['scaled'] = scaler.transform((data.passengers)) #np.log
for lag in range(1,25):
    data["lag%d" % lag] = np.hstack([np.nan]*lag + [data.scaled[:-lag].values])

def predict_nstep(model,X):
    n       = X.shape[0]
    max_lag = X.shape[2]
    y_hat   = np.vstack([X[0,:,:].T,[[np.nan]]*n])
    for t in range(max_lag,max_lag+n):
        y_hat[t] = model.predict(y_hat[t-max_lag:t].T[None,:,:])
    return y_hat[max_lag:]

def predict_nstep2(model,X):
    n       = X.shape[0]
    max_lag = X.shape[1]
    y_hat   = np.vstack([X[0,:,:],[[np.nan]]*n])
    for t in range(max_lag,max_lag+n):
        y_hat[t] = model.predict(y_hat[t-max_lag:t][None,:,:])
    return y_hat[max_lag:]

def predict_nstep3(model,X):
    n       = X.shape[0]
    max_lag = X.shape[1]
    y_hat   = np.vstack([X[[0],:].T,[[np.nan]]*n])
    for t in range(max_lag,max_lag+n):
        y_hat[t] = model.predict(y_hat[t-max_lag:t].T)
    return y_hat[max_lag:]

trainPredict = {}
testPredict = {}
testPredict_nstep = {}
trainScore  = {}
testScore   = {}
testScore_nstep  = {}
for max_lag in MAX_LAG_RANGE:
    col_lag = ["lag%d" % lag for lag in range(1,max_lag+1)]
    trainX  = data[max_lag:tr_size][col_lag].values
    trainY  = data[max_lag:tr_size].scaled
    testX   = data[tr_size:][col_lag].values
    testY   = data[tr_size:].scaled
    #
    np.random.seed(0) # fix random seed for reproducibility
    model  = Sequential()
    if MODEL_NAME == 'LSTM1':
        trainX  = data[max_lag:tr_size][col_lag].values[:,None,:]
        testX   = data[tr_size:][col_lag].values[:,None,:]
        model.add(LSTM(12, input_length=1, input_dim=max_lag)) #, activation='linear'
    elif MODEL_NAME == 'LSTM2':
        trainX  = data[max_lag:tr_size][col_lag].values[:,:,None]
        testX   = data[tr_size:][col_lag].values[:,:,None]
        model.add(LSTM(12, input_length=max_lag, input_dim=1))
    elif MODEL_NAME == 'LSTM3':
        trainX  = data[max_lag:tr_size][col_lag].values[None,:,:]
        testX   = data[tr_size:][col_lag].values[None,:,:]
        model.add(LSTM(12, input_length=tr_size-max_lag, input_dim=max_lag, return_sequences=True))
    elif MODEL_NAME == 'SimpleRNN1':
        trainX  = data[max_lag:tr_size][col_lag].values[:,None,:]
        testX   = data[tr_size:][col_lag].values[:,None,:]
        model.add(SimpleRNN(12, input_length=1, input_dim=max_lag)) #, activation='linear'
    elif MODEL_NAME == 'SimpleRNN2':
        trainX  = data[max_lag:tr_size][col_lag].values[:,:,None]
        testX   = data[tr_size:][col_lag].values[:,:,None]
        model.add(SimpleRNN(12, input_length=max_lag, input_dim=1))
    elif MODEL_NAME == 'MLP':
        model.add(Dense(12, input_dim=max_lag, activation='linear'))
    model.add(Dense(12, input_shape=(1,tr_size-max_lag,max_lag), activation='linear'))
    model.add(Dense(12, input_shape=(1,tr_size-max_lag,max_lag), activation='linear'))
    model.add(Dense(1, input_shape=(1,tr_size-max_lag,max_lag), activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    res  = model.fit(trainX, trainY, nb_epoch=200, batch_size=1, verbose=2)
    #
    trainPredict[max_lag] = (scaler.inverse_transform(model.predict(trainX))) #np.exp
    testPredict[max_lag]  = (scaler.inverse_transform(model.predict(testX))) #np.exp
    if MODEL_NAME in ('LSTM1','SimpleRNN1'):
        testPredict_nstep[max_lag]  = (scaler.inverse_transform(predict_nstep(model,testX))) #np.exp
    elif MODEL_NAME in ('LSTM2','SimpleRNN2'):
        testPredict_nstep[max_lag]  = (scaler.inverse_transform(predict_nstep2(model,testX))) #np.exp
    else:
        testPredict_nstep[max_lag]  = (scaler.inverse_transform(predict_nstep3(model,testX))) #np.exp
    #
    trainScore[max_lag] = np.sqrt(np.mean((data[max_lag:tr_size].passengers-trainPredict[max_lag].squeeze())**2))
    testScore[max_lag]  = np.sqrt(np.mean((data[tr_size:].passengers-testPredict[max_lag].squeeze())**2))
    testScore_nstep[max_lag]  = np.sqrt(np.mean((data[tr_size:].passengers-testPredict_nstep[max_lag].squeeze())**2))
    #
    plot(model, show_shapes=True, to_file=MODEL_NAME + '.png')
    #
    fig  = plt.figure(figsize=(16,10))
    plt.plot(time[tr_size-24:],y.squeeze()[tr_size-24:],'r:o',label='airline')
    plt.plot(time[tr_size:],y_hat_1step[tr_size:-1],'g:d',label="BSTSM 1-step RMSE = %.2f" % RMSE_tt_1step)
    plt.plot(time[tr_size:],y_hat[tr_size:],'g-d',label="BSTSM n-step RMSE = %.2f" % RMSE_tt)
    plt.plot(time[tr_size:],testPredict[max_lag],'b:s',label="%s 1-step RMSE = %.2f" % (MODEL_NAME,testScore[max_lag]))
    plt.plot(time[tr_size:],testPredict_nstep[max_lag],'b-s',label="%s n-step RMSE = %.2f" % (MODEL_NAME,testScore_nstep[max_lag]))
    plt.legend(loc='upper left')
    plt.savefig("airline_keras_%s_maxlag=%02d.png" % (MODEL_NAME,max_lag))
    plt.close()
    # plt.show()

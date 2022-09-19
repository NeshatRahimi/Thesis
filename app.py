from turtle import shape, shapesize
import netCDF4 as nc
import pandas as pd
import xlrd
from sklearn import svm
import numpy as np
import csv
from csv import reader
from csv import writer
import cftime
import xarray as xr
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from persiantools.jdatetime import JalaliDate
import jdatetime
from khayyam  import *
import datetime
from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Bidirectional, GRU, RepeatVector, Dense, TimeDistributed ,Activation
from sklearn.preprocessing import MinMaxScaler
from pyneurgen.neuralnet import NeuralNet
from pyneurgen.recurrent import NARXRecurrent
import tensorflow as tf
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.preprocessing import normalize
#model 1 preprocessing input data  havashenasi abdehi & mr rezaee data run once only for making excel file
"""path_pol_arvan = "/Users/neshat/Downloads/4106,Dr_modiri/mr.rezaee qazvin/abdehi_polArvan.csv"
data_havashenasi = pd.read_csv(path_pol_arvan)
print(data_havashenasi)

path_mr_rezaee = "/Users/neshat/Downloads/4106,Dr_modiri/mr.rezaee qazvin/Database for data driven models to rapidminer.xlsx"
data_rezaee = pd.read_excel(path_mr_rezaee)
pd.to_datetime(data_rezaee["Date"])

n = len(data_rezaee['Date'])
data_rezaee['time']= ""
print(data_rezaee)
for i in range(n):
    data_rezaee['time'][i] = jdatetime.date.fromgregorian(
        day=data_rezaee['Date'][i].day, month=data_rezaee['Date'][i].month, year=data_rezaee['Date'][i].year)
print(max(data_rezaee['time']))
print(min(data_havashenasi['sal']))

for j in range(n):
    for k in range(n):
        if data_rezaee['time'][j].day == data_havashenasi['rooz'][k] and data_rezaee['time'][j].month == data_havashenasi['mah'][k] and data_rezaee['time'][j].year == data_havashenasi['sal'][k]:
            data_rezaee['FLOW'][j]= data_havashenasi['abdehi'][k]

print(data_rezaee)
print(data_rezaee['Date'])
data_rezaee.to_excel('Qazvin_abdehi.xlsx')"""


"""data_havashenasi['tarikh'] = ""
data_havashenasi['tarikh'] = data_havashenasi[['sal', 'mah', 'rooz']].apply(pd.to_datetime)
print(data_havashenasi)"""

# model1 data Qazvinabdehi to svr 
"""path_qazvin_abdehi = '/Users/neshat/workspace/test/Qazvin_abdehi.xlsx'
opening = pd.read_excel(path_qazvin_abdehi)
d = opening.dropna()
abdehi_train = d.loc[ (d['Date'].dt.year < 2000)]
abdehi_test = d.loc[(d['Date'].dt.year > 1999)]

mah = d['Date'].dt.month
abdehi_train['mah'] = mah
abdehi_test['mah'] = mah

model1 = SVR(kernel='rbf', C=10, epsilon=0.001)

fitt = model1.fit(abdehi_train[' Air Temperature in Fahrenheit'].values.reshape(-1, 1), abdehi_train['FLOW'])
predictt = model1.predict(abdehi_test[' Air Temperature in Fahrenheit'].values.reshape(-1, 1))

fitpr = model1.fit(abdehi_train['h22'].values.reshape(-1, 1), abdehi_train['FLOW'])
predictpr = model1.predict(abdehi_test['h22'].values.reshape(-1, 1))

featuresf_test = abdehi_test[[' Air Temperature in Fahrenheit', 'h22']].values.tolist()
featuresf_train = abdehi_train[[' Air Temperature in Fahrenheit', 'h22']].values.tolist()
fitf = model1.fit(featuresf_train, abdehi_train['FLOW'])
predictf = model1.predict(featuresf_test)

print(abdehi_train) # make the dataframe with mah column the data frame has Date
fet_test = abdehi_test[[' Air Temperature in Fahrenheit', 'h22','mah']].values.tolist()
fet_train = abdehi_train[[' Air Temperature in Fahrenheit', 'h22','mah']].values.tolist()
fif = model1.fit(fet_train, abdehi_train['FLOW'])
pf = model1.predict(fet_test)


# make the dataframe with mah column the data frame has Date
# in LSTM are we predicting correctly?



# results Correlation coefficient calculation & Calculate RMSE
np.seterr(invalid='ignore')
rbf_corr = np.corrcoef(predictt, abdehi_test['FLOW'])[0, 1]
rbf_rmse = sqrt(mean_squared_error(predictt, abdehi_test['FLOW']))
print("hindcast model1 temperature GCM input SVR, RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))

rbf_corrpr = np.corrcoef(predictpr, abdehi_test['FLOW'])[0, 1]
rbf_rmsepr = sqrt(mean_squared_error(predictpr, abdehi_test['FLOW']))
print("hindcast model1 precipitation GCM input SVR, RBF: RMSE %f \t\t Corr %f" %(rbf_rmsepr, rbf_corrpr))

rbf_corrf = np.corrcoef(predictf, abdehi_test['FLOW'])[0, 1]
rbf_rmsef = sqrt(mean_squared_error(predictf, abdehi_test['FLOW']))
print("hindcast model1 both precipitation & temperature GCM input SVR, RBF: RMSE %f \t\t Corr %f" %(rbf_rmsef, rbf_corrf))

rbf_corrm= np.corrcoef(pf, abdehi_test['FLOW'])[0, 1]
rbf_rmsem = sqrt(mean_squared_error(pf, abdehi_test['FLOW']))
print("hindcast model1 precipitation & temperature & month GCM input SVR, RBF: RMSE %f \t\t Corr %f" %(rbf_rmsem, rbf_corrm))

plt.figure(figsize=(16,9))
plt.plot(abdehi_test['Date'],abdehi_test['FLOW'])
plt.plot(abdehi_test['Date'],pf)
plt.title("SVR ,input:model1 ,data:GCM")
plt.legend(('Actual','predicted'))
plt.show()"""

#model 2 preprocessing input data havashenasi abdehi & mr mirzaee data run once only for making excel file
"""path_m_station = '/Users/neshat/Downloads/4106,Dr_modiri/mr.rezaee qazvin/Qazvin.xlsx'
data_mr_mirzaee = pd.read_excel(path_m_station)
path_pol_arvan = "/Users/neshat/Downloads/4106,Dr_modiri/mr.rezaee qazvin/abdehi_polArvan.csv"
data_havashenasi = pd.read_csv(path_pol_arvan)
data_mr_mirzaee['time'] = ""
n = len(data_mr_mirzaee['data'])
print(n)
for i in range(n):`
    data_mr_mirzaee['time'][i] = jdatetime.date.fromgregorian(
        day=data_mr_mirzaee['data'][i].day, month=data_mr_mirzaee['data'][i].month, year=data_mr_mirzaee['data'][i].year)
print(data_mr_mirzaee)
print(data_mr_mirzaee['time'][2].day)
#print(data_havashenasi)
df = data_mr_mirzaee
df['FLOW'] = df['FLOW'].fillna(0)
print(df)


for k in range(len(data_havashenasi)):
    for j in range(len(df)):
        if df['time'][j].year == data_havashenasi['sal'][k] and df['time'][j].month == data_havashenasi['mah'][k] and df['time'][j].day == data_havashenasi['rooz'][k]:
            df['FLOW'][j] = data_havashenasi['abdehi'][k]
            print(data_havashenasi['sal'][k])
            print(df['time'][j].year)
            print(df['time'][j].month)
            print(df['time'][j].day)
            print(data_havashenasi['rooz'][k])
            print(data_havashenasi['mah'][k])
            print(data_havashenasi['abdehi'][k])
            print(df['FLOW'][j])
            print(k)   
            

print(df)
print(max(df['FLOW']))
df.to_excel('mr_mirzaee&abdehi.xlsx')"""

#####model 2 input main station SVR ####
"""path_qazvin_abdehi_mirzaee = '/Users/neshat/workspace/test/mr_mirzaee&abdehi.xlsx'
opening = pd.read_excel(path_qazvin_abdehi_mirzaee)
d = opening.dropna()
abdehi_train = d.loc[(d['data'].dt.year < 2000)]
abdehi_test = d.loc[(d['data'].dt.year > 1999)]

mah = d['data'].dt.month
abdehi_train['mah'] = mah
abdehi_test['mah'] = mah

model2 = SVR(kernel='rbf', C=49, epsilon=0.001)

fitt = model2.fit(abdehi_train['tm'].values.reshape(-1, 1), abdehi_train['FLOW'])
predictt = model2.predict(abdehi_test['tm'].values.reshape(-1, 1))

fitpr = model2.fit(abdehi_train['rrr24'].values.reshape(-1, 1), abdehi_train['FLOW'])
predictpr = model2.predict(abdehi_test['rrr24'].values.reshape(-1, 1))

featuresf_test = abdehi_test[['tm', 'rrr24']].values.tolist()
featuresf_train = abdehi_train[['tm', 'rrr24']].values.tolist()
fitf = model2.fit(featuresf_train, abdehi_train['FLOW'])
predictf = model2.predict(featuresf_test)

fef_test = abdehi_test[['tm', 'rrr24','mah']]#.values.tolist()
fef_train = abdehi_train[['tm', 'rrr24','mah']]#.values.tolist()
fitf = model2.fit(fef_train, abdehi_train['FLOW'])
pf = model2.predict(fef_test)

# results Correlation coefficient calculation & Calculate RMSE
np.seterr(invalid='ignore')
rbf_corr = np.corrcoef(predictt, abdehi_test['FLOW'])[0, 1]
rbf_rmse = sqrt(mean_squared_error(predictt, abdehi_test['FLOW']))
print("hindcast model2 temperature main station input SVR, RBF: RMSE %f \t\t Corr %f" %
      (rbf_rmse, rbf_corr))

rbf_corrpr = np.corrcoef(predictpr, abdehi_test['FLOW'])[0, 1]
rbf_rmsepr = sqrt(mean_squared_error(predictpr, abdehi_test['FLOW']))
print("hindcast model2 precipitation main station input SVR, RBF: RMSE %f \t\t Corr %f" %
      (rbf_rmsepr, rbf_corrpr))

rbf_corrf = np.corrcoef(predictf, abdehi_test['FLOW'])[0, 1]
rbf_rmsef = sqrt(mean_squared_error(predictf, abdehi_test['FLOW']))
print("hindcast model2 both precipitation & temperature main station input SVR, RBF: RMSE %f \t\t Corr %f" % (rbf_rmsef, rbf_corrf))

rbf_corrm = np.corrcoef(pf, abdehi_test['FLOW'])[0, 1]
rbf_rmsem = sqrt(mean_squared_error(pf, abdehi_test['FLOW']))
print("hindcast model2 precipitation & temperature & month main station input SVR, RBF: RMSE %f \t\t Corr %f" % (rbf_rmsem, rbf_corrm))


plt.figure(figsize=(16,9))
plt.plot(abdehi_test['data'],abdehi_test['FLOW'])
plt.plot(abdehi_test['data'],pf)
plt.title("SVR ,input:model2 ,data:main station")
plt.legend(('Actual','predicted'))
plt.show()"""

#adbedi timeseries LSTM algorithm 
"""path_qazvin_abdehi = '/Users/neshat/workspace/test/Qazvin_abdehi.xlsx'
opening = pd.read_excel(path_qazvin_abdehi)
d = opening.dropna()
abdehi_train = d.loc[(d['Date'].dt.year < 2000)]['FLOW']
abdehi_test = d.loc[(d['Date'].dt.year > 1999)]['FLOW']
abdehi_train_value = abdehi_train.values
abdehi_test_value = abdehi_test.values

#scaler = MinMaxScaler(feature_range=(0, 1))
#abdehi_train_value = scaler.fit_transform(abdehi_train_value.reshape(1, -1))
#abdehi_test_value = scaler.fit_transform(abdehi_test_value.reshape(1, -1))
print(abdehi_train_value)

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back)]
		dataX.append(a)
		dataY.append(dataset[i + look_back])
	return np.array(dataX), np.array(dataY)

look_back = 1
abdehi_trainX, abdehi_trainY = create_dataset(abdehi_train_value, look_back)
abdehi_testX, abdehi_testY = create_dataset(abdehi_test_value, look_back)
print(abdehi_trainX)
print(abdehi_trainY)

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(abdehi_trainX, abdehi_trainY, epochs=10, batch_size=1, verbose=2)



# make predictions
trainPredict = model.predict(abdehi_trainX)
testPredict = model.predict(abdehi_testX)
print(trainPredict)

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(abdehi_trainY, trainPredict[:]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(abdehi_testY, testPredict[:]))
print('Test Score: %.2f RMSE' % (testScore))

#corrf = np.corrcoef(abdehi_testY, testPredict.values)
#print('Correlation Coefficient: %.2f corr' % (corrf))"""


##### LSTM algorithm with exogenous variables model1 data Qazvinabdehi GCM to GCM ####

"""path_qazvin_abdehi = '/Users/neshat/workspace/test/Qazvin_abdehi.xlsx'
opening = pd.read_excel(path_qazvin_abdehi)
d = opening.dropna()
abdehi_train = d.loc[(d['Date'].dt.year < 2000)]
abdehi_test = d.loc[(d['Date'].dt.year > 1999)]
print(abdehi_train)
mah = d['Date'].dt.month
ruz = d['Date'].dt.day

abdehi_train['day'] = ruz
abdehi_train['month'] = mah


abdehi_test['day'] = ruz
abdehi_test['month'] = mah

print(abdehi_test)


for i in abdehi_train.select_dtypes('object').columns:
    le = LabelEncoder().fit(abdehi_train[i])
    abdehi_train[i] = le.transform(abdehi_train[i]) 
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    #X_data = X_scaler.fit_transform(abdehi_train[[' Air Temperature in Fahrenheit', 'h22','day','month','FLOW']])
    #Y_data =Y_scaler.fit_transform( abdehi_train[['FLOW']])
    X_data = abdehi_train[[' Air Temperature in Fahrenheit', 'h22','month','FLOW']]
    Y_data = abdehi_train[['FLOW']]

#print(f'y-data{Y_data.tolist()}')

flow=[]
for myList in Y_data:
    for item in myList:
        flow.append(item)

print(flow)

for i in abdehi_test.select_dtypes('object').columns:
    le = LabelEncoder().fit(abdehi_test[i])
    abdehi_test[i] = le.transform(abdehi_test[i]) 
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    #X_data = X_scaler.fit_transform(abdehi_train[[' Air Temperature in Fahrenheit', 'h22','day','month','FLOW']])
    #Y_data =Y_scaler.fit_transform( abdehi_train[['FLOW']])
    Xt_data = abdehi_test[[' Air Temperature in Fahrenheit', 'h22','month','FLOW']]
    Yt_data = abdehi_test[['FLOW']]


def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
     X = []
     y = []
     start = start + window
     if end is None:
         end = len(dataset) - horizon
     for i in range(start, end):
        #  indices = range(i-window, i)
         X.append(dataset[i-window:i])
        #  indicey = range(i+1, i+1+horizon)
         y.append(target[i+1: i+1+horizon])

        #
         #indices = range(i-window, i)
         #X.append(dataset[indices])
         #indicey = range(i+1, i+1+horizon)
         #y.append(target[indicey])
     return np.array(X), np.array(y)


hist_window = 5  #see last 5 days 
horizon = 1   #forecast the next day
TRAIN_SPLIT = 2000
x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon) 

print ('Multiple window of past history\n')
print(x_train[-2:]) #[0])
print ('\n Target horizon\n')
print (y_train)#[0]) 

batch_size = 100  #the number of values that we can use at a time to train
buffer_size = 50
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
val_data = val_data.batch(batch_size).repeat()

lstm_model = tf.keras.models.Sequential([
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True), 
                                input_shape=x_train.shape[-2:]),
     tf.keras.layers.Dense(20, activation='relu'),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
     tf.keras.layers.Dense(20, activation='relu'),
     tf.keras.layers.Dense(20, activation='relu'),
     tf.keras.layers.Dropout(0.25),
     tf.keras.layers.Dense(units=horizon),
 ])
lstm_model.compile(optimizer='adam', loss='mse')

model_path = 'Bidirectional_LSTM_Multivariate.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint] 
history = lstm_model.fit(train_data,epochs=70,steps_per_epoch=100,validation_data=val_data,validation_steps=70,verbose=1,callbacks=callbacks)
print(train_data)
print(val_data)
plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for LSTM ,input:model1 ,data: GCM')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
#plt.show()

pred=[]
for i in range(2368):  #1190 row  #2368 Y_true
    #data_val = X_scaler.fit_transform(abdehi_train[[' Air Temperature in Fahrenheit', 'h22','month','FLOW']][i:i+5]).values  #tail means take the last 5 
    data_val = abdehi_train[[' Air Temperature in Fahrenheit', 'h22','month','FLOW']][i:i+5].values
    val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
    pre = lstm_model.predict(val_rescaled)
    pred_Inverse = pre
    pred.append(pred_Inverse.item())#Y_scaler.inverse_transform(pre)

    #print(f'pred_Inverse{pred_Inverse}')
    #print(f'appending{pred.append(pred_Inverse.item())}')
    #print(f'pred in loop {pred}')

#print(f'pred out of loop {pred}')
pr = np.array(pred)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = pr.reshape(-1, 1)
#X_train_minmax = pr.reshape(-1, 1).values
#X_train_minmax = min_max_scaler.fit_transform(pr.reshape(-1, 1))
#print(f'X_train_minmax : {X_train_minmax}')

def timeseries_evaluation_metrics_func(y_true, y_pred):
     def mean_absolute_percentage_error(y_true, y_pred): 
         y_true, y_pred = np.array(y_true), np.array(y_pred)
         return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     print('Evaluation metric results:-')
     #print(y_true.shape, y_pred.shape)
     #print(y_pred)
     #print(y_true)
     print(f'MSE for LSTM model1 GCM is : {metrics.mean_squared_error(y_true, y_pred)}')
     print(f'MAE for LSTM model1 GCM is : {metrics.mean_absolute_error(y_true, y_pred)}')
     print(f'RMSE for LSTM model1 GCM is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
     print(f'MAPE for LSTM model1 GCM is : {mean_absolute_percentage_error(y_true, y_pred)}')
     print(f'Corr for LSTM model1 GCM is : {np.corrcoef(y_true, y_pred)[0, 1]}')


pr = np.array(pred)
X_train_minmax = pr.reshape(-1, 1)
timeseries_evaluation_metrics_func(Y_data, X_train_minmax)
#timeseries_evaluation_metrics_func(np.array(abdehi_train['FLOW'])[-1].reshape(1, 1), pred_Inverse)

#timeseries_evaluation_metrics_func(Y_data, X_train_minmax)

plt.figure(figsize=(16,9))
plt.plot(abdehi_train['Date'],Y_data)
plt.plot(abdehi_train['Date'],list(X_train_minmax))
plt.title("LSTM ,input:model1 ,data: GCM")
plt.legend(('Actual','predicted'))
plt.show()"""



##### LSTM algorithm with exogenous variables model2 data Qazvinabdehi main station ####
path_qazvin_abdehi_mirzaee = '/Users/neshat/workspace/test/mr_mirzaee&abdehi.xlsx'
opening = pd.read_excel(path_qazvin_abdehi_mirzaee)
d = opening.dropna()
abdehi_train = d.loc[(d['data'].dt.year < 2000)]
abdehi_test = d.loc[(d['data'].dt.year > 1999)]

print(abdehi_train)

mah = d['data'].dt.month
ruz = d['data'].dt.day

abdehi_train['day'] = ruz
abdehi_train['month'] = mah


abdehi_test['day'] = ruz
abdehi_test['month'] = mah

print(abdehi_test)


for i in abdehi_train.select_dtypes('object').columns:
    le = LabelEncoder().fit(abdehi_train[i])
    abdehi_train[i] = le.transform(abdehi_train[i]) 
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    #X_data = X_scaler.fit_transform(abdehi_train[[' Air Temperature in Fahrenheit', 'h22','day','month','FLOW']])
    #Y_data =Y_scaler.fit_transform( abdehi_train[['FLOW']])
    X_data = abdehi_train[['tm', 'rrr24','month','FLOW']]
    Y_data = abdehi_train[['FLOW']]

#print(f'y-data{Y_data.tolist()}')

flow=[]
for myList in Y_data:
    for item in myList:
        flow.append(item)

print(flow)

for i in abdehi_test.select_dtypes('object').columns:
    le = LabelEncoder().fit(abdehi_test[i])
    abdehi_test[i] = le.transform(abdehi_test[i]) 
    X_scaler = MinMaxScaler()
    Y_scaler = MinMaxScaler()
    #X_data = X_scaler.fit_transform(abdehi_train[[' Air Temperature in Fahrenheit', 'h22','day','month','FLOW']])
    #Y_data =Y_scaler.fit_transform( abdehi_train[['FLOW']])
    Xt_data = abdehi_test[['tm', 'rrr24','month','FLOW']]
    Yt_data = abdehi_test[['FLOW']]


def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
     X = []
     y = []
     start = start + window
     if end is None:
         end = len(dataset) - horizon
     for i in range(start, end):
        #  indices = range(i-window, i)
         X.append(dataset[i-window:i])
        #  indicey = range(i+1, i+1+horizon)
         y.append(target[i+1: i+1+horizon])

        #
         #indices = range(i-window, i)
         #X.append(dataset[indices])
         #indicey = range(i+1, i+1+horizon)
         #y.append(target[indicey])
     return np.array(X), np.array(y)


hist_window = 5  #see last 5 days 
horizon = 1   #forecast the next day
TRAIN_SPLIT = 2000
x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon) 

print ('Multiple window of past history\n')
print(x_train[-2:]) #[0])
print ('\n Target horizon\n')
print (y_train)#[0]) 

batch_size = 100  #the number of values that we can use at a time to train
buffer_size = 50
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(buffer_size).batch(batch_size).repeat()
val_data = tf.data.Dataset.from_tensor_slices((x_vali, y_vali))
val_data = val_data.batch(batch_size).repeat()

lstm_model = tf.keras.models.Sequential([
   tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(200, return_sequences=True), 
                                input_shape=x_train.shape[-2:]),
     tf.keras.layers.Dense(20, activation='relu'),
     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
     tf.keras.layers.Dense(20, activation='relu'),
     tf.keras.layers.Dense(20, activation='relu'),
     tf.keras.layers.Dropout(0.25),
     tf.keras.layers.Dense(units=horizon),
 ])
lstm_model.compile(optimizer='adam', loss='mse')

model_path = 'Bidirectional_LSTM_Multivariate.h5'
early_stopings = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='min')
checkpoint =  tf.keras.callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=0)
callbacks=[early_stopings,checkpoint] 
history = lstm_model.fit(train_data,epochs=70,steps_per_epoch=100,validation_data=val_data,validation_steps=70,verbose=1,callbacks=callbacks)
print(train_data)
print(val_data)
plt.figure(figsize=(16,9))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss for LSTM ,input: model2 ,data: main station')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'validation loss'])
#plt.show()

pred=[]
for i in range(6500):  #6500 row  #7304 Y_true
    #data_val = X_scaler.fit_transform(abdehi_train[[' Air Temperature in Fahrenheit', 'h22','month','FLOW']][i:i+5]).values  #tail means take the last 5 
    data_val = abdehi_train[['tm', 'rrr24','month','FLOW']][i:i+5].values
    val_rescaled = data_val.reshape(1, data_val.shape[0], data_val.shape[1])
    pre = lstm_model.predict(val_rescaled)
    pred_Inverse = pre
    pred.append(pred_Inverse.item())#Y_scaler.inverse_transform(pre)

    #print(f'pred_Inverse{pred_Inverse}')
    #print(f'appending{pred.append(pred_Inverse.item())}')
    #print(f'pred in loop {pred}')

#print(f'pred out of loop {pred}')
pr = np.array(pred)
min_max_scaler = preprocessing.MinMaxScaler()
X_train_minmax = pr.reshape(-1, 1)
#X_train_minmax = pr.reshape(-1, 1).values
#X_train_minmax = min_max_scaler.fit_transform(pr.reshape(-1, 1))
#print(f'X_train_minmax : {X_train_minmax}')

def timeseries_evaluation_metrics_func(y_true, y_pred):
     def mean_absolute_percentage_error(y_true, y_pred): 
         y_true, y_pred = np.array(y_true), np.array(y_pred)
         return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
     print('Evaluation metric results:-')
     #print(y_true.shape, y_pred.shape)
     #print(y_pred)
     print(y_true)
     print(f'MSE for LSTM model2 main station is : {metrics.mean_squared_error(y_true, y_pred)}')
     print(f'MAE for LSTM model2 main station is : {metrics.mean_absolute_error(y_true, y_pred)}')
     print(f'RMSE for LSTM model2 main station is : {np.sqrt(metrics.mean_squared_error(y_true, y_pred))}')
     print(f'MAPE for LSTM model2 main station is : {mean_absolute_percentage_error(y_true, y_pred)}')
     print(f'Corr for LSTM model2 main station is : {np.corrcoef(y_true, y_pred)[0, 1]}')


pr = np.array(pred)
X_train_minmax = pr.reshape(-1, 1)
timeseries_evaluation_metrics_func(Y_data, X_train_minmax)
#timeseries_evaluation_metrics_func(np.array(abdehi_train['FLOW'])[-1].reshape(1, 1), pred_Inverse)

#timeseries_evaluation_metrics_func(Y_data, X_train_minmax)

plt.figure(figsize=(16,9))
plt.plot(abdehi_train['data'],Y_data)
plt.plot(abdehi_train['data'],list(X_train_minmax))
plt.title("LSTM ,input: model2 ,data: main station")
plt.legend(('Actual','predicted'))
plt.show()





############################################################################

### chert o pert ###

# gcm 1975 - 2000 precipitation 

#GCM data precipitation daily to csv run once only
"""path_gcm1 = '/Users/neshat/Downloads/GCM/pr_day_ACCESS1-0_historical_r3i1p1_19750101-19991231.nc'
opening = xr.open_dataset(path_gcm1)
pr1 = opening.pr.to_dataframe().reset_index()

pr1_sliced = pr1[(pr1.lat > 34) & (pr1.lat < 37) & (pr1.lon > 47) & (pr1.lon < 51)]
print(pr1_sliced)
pr1_sliced.to_csv('precipgcmiran.csv')"""

#downscaled data Qazvin 
"""path_mr_mirzae = '/Users/neshat/Downloads/Qazvin.xlsx'
open_ds = pd.read_excel(path_mr_mirzae)

t = pd.to_datetime(open_ds['data'])

#GCM data ACCESS precipitation & runoff 1990 - 2005 rbf prepairation
open_ds['date'] = t.dt.month
open_ds['year'] = t.dt.year

df = open_ds.set_index(['lat', 'lon', 'year', 'date'])
jam = df.groupby(level=['lat', 'lon', 'year', 'date']).sum('rrr24')
#jam = open_ds.reset_index(['lat', 'lon', 'year', 'date'])
print(jam.year)"""


#jalili_date = jdatetime.date(jam.year, jam.month, jam.rooz).togregorian()
#print(jalili_date)

#global station data precipitation 1990-1991
"""path_gcm = '/Users/neshat/Downloads/4106,Dr_modiri/1990_To_1991_baniardalan.xlsx'
opening = pd.read_excel(path_gcm)
print(opening)


dataset_pr_gcm = pd.read_csv('precipgcmiran.csv')

t = pd.to_datetime(opening['utc'])
tgcm = pd.to_datetime(dataset_pr_gcm['time'])

#1990 and rbf

#GCM data 1990 rbf
year1990gcm = dataset_pr_gcm.loc[tgcm.dt.year == 1990]
precip1990gcm = round(year1990gcm.pr*100000,3)
lat1990gcm = year1990gcm.lat
lon1990gcm = year1990gcm.lon
month1990gcm = pd.to_datetime(year1990gcm['time']).dt.month
features1990gcm = np.c_[month1990gcm, lat1990gcm,lon1990gcm]  # set kernel and hyperparameters
#print(precip1990gcm)
for n in range(1, 13):
    jam = sum(precip1990gcm.loc[tgcm.dt.month == n])
    precip1990gcm.loc[tgcm.dt.month == n] = jam*1000
print(precip1990gcm)
print("max pr 1990 gcm (mm3) : %f" % max(precip1990gcm))
print(lat1990gcm)
#stations data 1990 rbf
year1990 = opening.loc[t.dt.year == 1990]
print(year1990)
precip1990 = year1990.precip # mm3 to cm3
lat1990 = round(year1990.latitude)
lon1990 = round(year1990.longitude)
month1990 = pd.to_datetime(year1990['utc']).dt.month
features1990 = np.c_[month1990, lat1990, lon1990]
print("max pr 1990 station ( mm3): %f" % max(precip1990))
print(lat1990)
#GCM train & station test
model1990 = SVR(kernel='rbf', C=1, epsilon=10)
fit1990 = model1990.fit(
    features1990gcm, precip1990gcm).predict(features1990gcm)
predict1990 = model1990.predict(features1990)

# results for year 1990 Correlation coefficient calculation & Calculate RMSE
np.seterr(invalid='ignore')
rbf_corr = np.corrcoef(precip1990, predict1990)[0, 1]
rbf_rmse = sqrt(mean_squared_error(precip1990, predict1990))
print("GCM to station RBF1990: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))


#station train & GCM test
model1990 = SVR(kernel='rbf', C=1, epsilon=10)
fit1990 = model1990.fit(features1990, precip1990).predict(features1990)
predict1990 = model1990.predict(features1990gcm)
#print(t.dt.year)

# results for year 1990 Correlation coefficient calculation & Calculate RMSE
np.seterr(invalid='ignore')
rbf_corr = np.corrcoef(precip1990gcm, predict1990)[0, 1]
rbf_rmse = sqrt(mean_squared_error(precip1990gcm, predict1990))
print("station to GCM RBF1990: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))"""

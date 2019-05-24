#!/env/Scripts python
# @File    : predict.py
# @Time    : 2019/4/9 10:41
# @Author  : Bb
# -*- coding: utf-8 -*-

from math import sqrt
from reader import XLSReader
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM,Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


reader = XLSReader()
df_result = reader.read()
# print(df_result)
print(df_result)

# nan to mean
df_result_re = df_result.fillna(df_result.mean()[:3])
# print(df_result.fillna(df_result.mean()[:3]).loc[df_result['y_pred'] == 1])


values = df_result_re.values
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


current_X = scaled[:-1,:3]
# reshape input to be 3D(samples,timesteps,features)
current_X = current_X.reshape(current_X.shape[0],1,current_X.shape[1])


#download model
model = load_model('currentlstm.h5')

plt.figure(figsize=(24,12))
current_predict = model.predict(current_X)
plt.plot(scaled[:,0], c='b')
plt.plot([x for x in current_predict], c='r')
plt.show()

# 归一化反转
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
# make a prediction
yhat = current_predict
test_X = scaled[:-1,:2]

# invert scaling for forecast
inv_yhat = concatenate((yhat,test_X), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# 原数据与标准化预测值对比图
plt.plot(df_result.iloc[1:, 0].values, c='b')
# plt.plot(values[:-1,0], c='b')
plt.plot([None for _ in df_result.iloc[1:450, 0].values] + [x for x in inv_yhat[450:]], c='r')
# plt.plot(inv_yhat[451:], c='r')
plt.show()


# calculate RMSE
rmse = sqrt(mean_squared_error(values[1:,0], inv_yhat[:]))
print('Test RMSE: %.3f' % rmse)




#!/env/Scripts python
# @File    : testdemo.py
# @Time    : 2019/5/22 14:17
# @Author  : Bb
# -*- coding: utf-8 -*-

from reader import XLSReader
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

reader = XLSReader()
df_result = reader.read()

# nan to mean
df_result_re = df_result.fillna(df_result.mean()[:3])
df_result_last = df_result_re.iloc[-1:]
print(df_result_last)

values = df_result_last.values
# ensure all data is float
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

current_X = scaled[-1:,:3]
current_predict = values

for i in range(12):
    # reshape input to be 3D(samples,timesteps,features)
    current_X = current_X.reshape(current_X.shape[0],1,current_X.shape[1])

    #download model
    model = load_model('currentlstm.h5')
    currentA_predict = model.predict(current_X)
    # print(currentA_predict)

    model = load_model('currentB_lstm.h5')
    currentB_predict = model.predict(current_X)
    # print(currentB_predict)

    model = load_model('currentC_lstm.h5')
    currentC_predict = model.predict(current_X)
    # print(currentC_predict)

    # 归一化反转
    from numpy import concatenate
    # invert scaling for forecast
    inv_yhat = concatenate((currentA_predict,currentB_predict,currentC_predict), axis=1)
    current_X = inv_yhat
    inv_yhat = scaler.inverse_transform(inv_yhat)
    # print(inv_yhat)
    current_predict = np.row_stack((current_predict, inv_yhat))
    # print("%d:"%i,current_X)
    print('数据处理中: {:.2%}'.format((i+1) /12))


#current_predict draw
for i in range(3):
    plt.figure(figsize=(24,12))
    plt.plot(df_result_re.iloc[:,i], c='b')
    plt.plot([None for _ in df_result_re.iloc[:,i]] + [x for x in current_predict[1:,i]], c='r')
    plt.show()

print('done!')
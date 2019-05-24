#!/env/Scripts python
# @File    : train.py
# @Time    : 2019/4/4 15:56
# @Author  : Bb
# -*- coding: utf-8 -*-

from reader import XLSReader
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import LSTM,Dense
from sklearn.preprocessing import MinMaxScaler

reader = XLSReader()
df_result = reader.read()
print(df_result)

# print(df_result)
# plt.plot(df_result.iloc[:, :3])
# plt.show()
df_result = df_result.fillna(df_result.mean()[:3])
# print(df_result.fillna(df_result.mean()[:3]).loc[df_result['y_pred'] == 1])


values = df_result.values
# ensure all data is float
values = values.astype('float32')
# nan to mean
# print(np.mean(values[:,0]))
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)


# split into train and test sets
splitpoint = 450
train = scaled[:splitpoint,:]
test = scaled[splitpoint:,:]
# split into input and output
train_X,train_y = train[:-1,:3],train[1:,0]
test_X, test_y = test[:-1, :3], test[1:, 0]
# reshape input to be 3D(samples,timesteps,features)
train_X = train_X.reshape(train_X.shape[0],1,train_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
# print(self.train_X.shape,self.train_y.shape,test_X.shape,test_y.shape)




# design network

model = Sequential()
# input_shape = (time_step, 每个时间步的input_dim)
# LSTM的第一个参数5表示LSTM的单元数为5，我们可以把LSTM理解为一个特殊的且带有时序信息的全连接层。
# model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(4, input_shape=(train_X.shape[1], train_X.shape[2]),return_sequences=True))
model.add(LSTM(4, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, nb_epoch=20, batch_size=1,validation_data=(test_X, test_y), verbose=2,shuffle=False)
model.save("currentlstm.h5")
#plot history
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.figure(figsize=(16,8))
train_predict = model.predict(train_X)
test_predict = model.predict(test_X)
plt.plot(scaled[1:,0], c='b')
plt.plot([x for x in train_predict], c='g')
plt.plot([None for _ in train_predict] + [x for x in test_predict], c='y')
plt.show()
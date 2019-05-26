import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from tools import save_info, show_plot
from tools import mean_absolute_percentage_error
from scipy.stats.stats import pearsonr
import time
from sklearn.preprocessing import MinMaxScaler


np.random.seed(2018)


class DNN(object):
    def __init__(self, X, bs, ts):
        self.NB_EPOCH = 200
        self.BATCH_SIZE = 500
        self.VERBOSE = 1
        self.OPTIMIZER = Adam()
        self.N_HIDDEN = 512
        self.VALIDATION_SPLIT = 0.14
        self.INPUT_DIM = int(bs)

        self.bs = bs
        self.ts = ts

        self.WINDOW_SIZE = int(bs)

        X, Y = self.prepare_data(X)
        Y = np.array(Y)

        Y = Y.reshape(-1, 1)

        scaler_X = MinMaxScaler()
        scaler_X = scaler_X.fit(X)
        X = scaler_X.transform(X)

        scaler_Y = MinMaxScaler()
        scaler_Y = scaler_Y.fit(Y)
        Y = scaler_Y.transform(Y)

        self.scaler_X = scaler_X
        self.scaler_Y = scaler_Y

        X = X[len(X) - ts:]
        Y = Y[len(Y) - ts:]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.1)

        self.X = X
        self.Y = Y
        self.X_train = X_train
        self.y_train = Y_train
        self.X_test = X_test
        self.y_test = Y_test

        print(X_train.shape)

    def prepare_data(self, X):
        series = pd.Series(X)
        series_s = series.copy()

        for i in range(self.WINDOW_SIZE):
            series = pd.concat([series, series_s.shift(-(i+1))], axis=1)
        series.dropna(axis=0, inplace=True)
        series.columns = np.arange(self.WINDOW_SIZE + 1)

        X_new = pd.DataFrame()
        for i in range(self.WINDOW_SIZE):
            X_new[i] = series[i]
        Y_new = series[self.WINDOW_SIZE]

        return X_new, Y_new

    def dnn(self, path, name):
        model = Sequential()
        print(len(self.X_train))
        model.add(Dense(self.INPUT_DIM, input_shape=(self.INPUT_DIM,)))
        model.add(Activation('relu'))
        for i in range(7):
            model.add(Dense(self.N_HIDDEN))
            model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('linear'))
        model.summary()
        model.compile(loss='mse',
                      optimizer=self.OPTIMIZER,
                      metrics=['accuracy'])
        history = model.fit(self.X_train, self.y_train,
                            epochs=self.NB_EPOCH,
                            verbose=self.VERBOSE)

        y_predict = model.predict(self.X_test)

        y_predict = y_predict.reshape(-1)

        mape_error = mean_absolute_percentage_error(self.y_test, y_predict)

        # save_info(self.y_test, y_predict, name, mape_error, self.WINDOW_SIZE, path, self.bs, self.ts)
        show_plot(self.y_test, y_predict)
        return model

    def lstm(self, path, name):
        trainX = np.reshape(self.X_train, (self.X_train.shape[0], 1, self.X_train.shape[1]))
        testX = np.reshape(self.X_test, (self.X_test.shape[0], 1, self.X_test.shape[1]))

        model = Sequential()
        model.add(LSTM(32, batch_input_shape=(1, trainX.shape[1], trainX.shape[2]), stateful=True))
        # model.add(Activation('tanh'))
        model.add(Dense(1))
        # model.add(Activation('linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        # model.summary()

        for i in range(20):
            model.fit(trainX, self.y_train, epochs=1, batch_size=1, verbose=self.VERBOSE, shuffle=False)
            model.reset_states()

        y_predict = model.predict(testX, batch_size=1)

        y_predict = y_predict.reshape(-1)

        mape_error = mean_absolute_percentage_error(self.y_test, y_predict)

        # save_info(self.y_test, y_predict, name, mape_error, self.WINDOW_SIZE, path, self.bs, self.ts)
        show_plot(self.y_test, y_predict)
        return model


#     def calculate_corr(self):
#         # corr = pearsonr(self.X.T, self.Y.T)
#         # np.savetxt('corrdata.txt', corr)
#         np.savetxt('X.txt', self.X)
#         np.savetxt('Y.txt', self.Y)


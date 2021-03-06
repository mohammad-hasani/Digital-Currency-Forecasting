from DNN import DNN
import numpy as np
from tools import *
from sklearn import preprocessing
import os
import time


def main():
    currencies = read_data()

    # data = [read_one_data('./Data/Bitcoin.csv')]

    bs = [i for i in range(50, 1000, 50)]
    bs = bs[:13]
    # ts2bs = [i * 2 for i in bs]
    ts3bs = [i * 3 for i in bs]
    # bs = bs * 2
    # ts = ts2bs + ts3bs
    ts = ts3bs
    bs = list([bs[1]])
    ts = list([ts[1]])
    print(bs)
    print(ts)
    # time.sleep(1000)
    # normalizer = preprocessing.Normalizer().fit([data])
    # data = normalizer.transform([data])
    # data = data.reshape(-1)

    for i in currencies:
        tmp = i[1]['Open']
        name = i[0].split('.')[0]
        path = './new/'

        # sort_data(path, name + '.txt')

        # if not os.path.exists(path + name):
        #     os.mkdir(path + name)
        for j in range(len(bs)):
            dnn = DNN(tmp, bs[j], ts[j])
            # model = dnn.dnn(path, name)
            new_name = ''
            # if j < len(bs) / 2:
            #     new_name = name + '2bs'
            # else:
            #     new_name = name + '3bs'
            lstm = dnn.dnn(path, new_name)
            # model.save('./new/' + name + '/dnn_' + str(name) + '_' + str(j) + '_' + '_model.h5')

    # import tools
    # for i in currencies:
    #     name = i[0].split('.')[0]
    #     tools.show_single_plot(i[1]['Open'], name)
    #
    # print(currencies[0][1]['Open'])

    # sort_bitcoin_data()

    # recognize_data()


if __name__ == '__main__':
    main()

# -*-coding=utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os
from datetime import datetime
PaviaU = '../data/PaviaU.mat'
PaviaU_gt = '../data/PaviaU_gt.mat'
Indian_pines = '../data/Indian_pines.mat'
Indian_pines_gt = '../data/Indian_pines_gt.mat'
Salinas = '../data/Salinas_corrected.mat'
Salinas_gt = '../data/Salinas_gt.mat'

for NO_data in ['Salians']:
    for sample_num in [200]:
        if NO_data == 'PaviaU':#use paviaU
            data = sio.loadmat(PaviaU)
            data_gt = sio.loadmat(PaviaU_gt)
            im = data['paviaU']
            imGIS = data_gt['paviaU_gt']
        elif NO_data == 'Inidan_pines':#use indian_pines
            data = sio.loadmat(Indian_pines)
            data_gt = sio.loadmat(Indian_pines_gt)
            im = data['indian_pines']
            imGIS = data_gt['indian_pines_gt']
            iG = np.zeros([imGIS.shape[0], imGIS.shape[1]], dtype=int)
            C = np.max(imGIS)
            origin_num = np.zeros(shape=[C + 1], dtype=int)
            for i in range(imGIS.shape[0]):
                for j in range(imGIS.shape[1]):
                    for k in range(1, C + 1):
                        if imGIS[i][j] == k:
                            origin_num[k] += 1
            index = 0
            data_num = np.zeros(shape=[9], dtype=int)
            data_label = np.zeros(shape=[9], dtype=int)
            for i in range(len(origin_num)):
                if origin_num[i] > 400:
                    data_num[index] = origin_num[i]
                    data_label[index] = i
                    index += 1
            # del too few classes
            for i in range(imGIS.shape[0]):
                for j in range(imGIS.shape[1]):
                    if imGIS[i, j] in data_label:
                        for k in range(len(data_label)):
                            if imGIS[i][j] == data_label[k]:
                                iG[i, j] = k + 1
                                continue
            imGIS = iG
        elif NO_data == 'Salians':#use Salinas
            data = sio.loadmat(Salinas)
            data_gt = sio.loadmat(Salinas_gt)
            im = data['salinas_corrected']
            imGIS = data_gt['salinas_gt']

        im = (im - float(np.min(im)))
        im = im/np.max(im)

        #sample_num = 200
        deepth = im.shape[2]
        classes = np.max(imGIS)

        data_ = {}
        train_data = {}
        test_data = {}

        for i in range(1,classes+1):
            data_[i]=[]
            train_data[i]=[]
            test_data[i] = []

        for i in range(imGIS.shape[0]):
            for j in range(imGIS.shape[1]):
                for k in range(1,classes+1):
                    if imGIS[i,j]==k:
                        data_[k].append(im[i,j])

        for i in range(1,classes+1):
            indexies = random.sample(range(len(data_[i])),sample_num)
            for k in range(len(data_[i])):
                if k not in indexies:
                    test_data[i].append(data_[i][k])
                else:
                    train_data[i].append(data_[i][k])

        train = []
        train_label = []
        test = []
        test_label = []

        for i in range(1,len(train_data)+1):
            for j in range(len(train_data[i])):
                train.append(train_data[i][j])
                train_label.append(i)

        for i in range(1,len(test_data)+1):
            for j in range(len(test_data[i])):
                test.append(test_data[i][j])
                test_label.append(i)
        if not os.path.exists('result'):
            os.makedirs('result')

        time = str(datetime.now().strftime('%Y%m%d_%H%M%S'))
        result = open('result/result_for_NO_data_'+str(NO_data)+'_'+'_samplenum_'+str(sample_num)+'_'+str(k)+'_'+time+'.txt','w')
        clf = KNeighborsClassifier(n_neighbors=3)
        clf.fit(train,train_label)
        C = np.max(imGIS)

        matrix = np.zeros((C,C))
        for i in range(len(test)):
            r = clf.predict(test[i].reshape(-1,len(test[i])))
            matrix[r-1,test_label[i]-1] += 1
        print(np.int_(matrix))
        print(np.sum(np.trace(matrix)))
        print(np.sum(np.trace(matrix)) / float(len(test_label)))
        result.write(str(np.int_(matrix)))
        result.write('\n')
        result.write(str(np.sum(np.trace(matrix))))
        result.write('\n')
        result.write(str(np.sum(np.trace(matrix)) / float(len(test_label))))
        result.write('\n')
        for i in range(len(matrix)):
            ac = matrix[i,i]/sum(matrix[:,i])
            print('accuracy ',i+1,':',ac)
            result.write(str(i+1)+":"+str(ac))
            result.write('\n')
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i])*np.sum(matrix[:,i])
        pe = kk/(np.sum(matrix)*np.sum(matrix))
        pa = np.trace(matrix)/np.sum(matrix)
        kappa = (pa-pe)/(1-pe)
        print('kappa:',kappa)
        result.write('kappa:'+str(kappa))
        result.write('\n')

        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        df_cm = pd.DataFrame(matrix, index = [i for i in alphabet[:matrix.shape[0]]],
                          columns = [i for i in alphabet[:matrix.shape[1]]])
        plt1 = plt
        plt1.figure(figsize = (matrix.shape[0],matrix.shape[1]))
        sn.heatmap(df_cm, annot=True,fmt="g")
        plt1.savefig('result/confusion_matrix1_'+'_samplenum_'+str(sample_num)+'_'+str(k)+'_'+time+'.png', format='png')
        plt1.close()

        iG = np.zeros((imGIS.shape[0],imGIS.shape[1]))
        for i in range(imGIS.shape[0]):
            for j in range(imGIS.shape[1]):
                if imGIS[i,j]==0:
                    iG[i,j]=0
                else:
                    iG[i,j] = (clf.predict(im[i,j].reshape(-1,len(im[i,j]))))
        plt2 = plt
        plt.pcolor(iG)
        plt.savefig('result/decode_result_'+'_samplenum_'+str(sample_num)+'_'+str(k)+'_'+str(time)+'.png', format='png')
        plt.close()

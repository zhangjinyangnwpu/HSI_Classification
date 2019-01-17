# -*-coding=utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import numpy as np
import scipy.io as sio
import random
import matplotlib.pyplot as plt
import os
path = os.path.join("E:\Code\RemoteDataProcessing\HSI_Classification\Data")# change this path for your dataset
PaviaU = os.path.join(path,'PaviaU.mat')
PaviaU_gt = os.path.join(path,'PaviaU_gt.mat')
Indian_pines = os.path.join(path,'Indian_pines.mat')
Indian_pines_gt = os.path.join(path,'Indian_pines_gt.mat')
Salinas = os.path.join(path,'Salinas.mat')
Salinas_gt = os.path.join(path,'Salinas_gt.mat')

NO_data = 'Salinas'

if NO_data == 'PaviaU':#use paviaU
    data = sio.loadmat(PaviaU)
    data_gt = sio.loadmat(PaviaU_gt)
    im = data['paviaU']
    imGIS = data_gt['paviaU_gt']
elif NO_data == 'Indian_pines':#use indian_pines
    data = sio.loadmat(Indian_pines)
    data_gt = sio.loadmat(Indian_pines_gt)
    im = data['indian_pines']
    imGIS = data_gt['indian_pines_gt']
elif NO_data == 'Salinas':#use Salinas
    data = sio.loadmat(Salinas)
    data_gt = sio.loadmat(Salinas_gt)
    im = data['salinas_corrected']
    imGIS = data_gt['salinas_gt']

im = (im - float(np.min(im)))
im = im/np.max(im)

sample_num = 200
neighuour_num = 5
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
    if NO_data == "Indian_pines" and len(data_[i]) < sample_num:
        sn = 15
    else:
        sn = sample_num
    indexies = random.sample(range(len(data_[i])),sn)
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
if not os.path.exists('./result'):
    os.makedirs('./result')
clf = KNeighborsClassifier(n_neighbors=neighuour_num)
train = np.asarray(train)
train_label = np.asarray(train_label)
clf.fit(train,train_label)
C = np.max(imGIS)

matrix = np.zeros((C,C))
for i in range(len(test)):
    r = clf.predict(test[i].reshape(-1,len(test[i])))
    matrix[r-1,test_label[i]-1] += 1


ac_list = []
for i in range(len(matrix)):
    ac = matrix[i, i] / sum(matrix[:, i])
    ac_list.append(ac)
    print(i+1,'class:','(', matrix[i, i], '/', sum(matrix[:, i]), ')', ac)
print('confusion matrix:')
print(np.int_(matrix))
print('total right num:', np.sum(np.trace(matrix)))
accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
print('oa:', accuracy)
# kappa
kk = 0
for i in range(matrix.shape[0]):
    kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
pe = kk / (np.sum(matrix) * np.sum(matrix))
pa = np.trace(matrix) / np.sum(matrix)
kappa = (pa - pe) / (1 - pe)
ac_list = np.asarray(ac_list)
aa = np.mean(ac_list)
oa = accuracy
print('aa:',aa)
print('kappa:', kappa)
sio.savemat(os.path.join('result', 'result.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix})
iG = np.zeros((imGIS.shape[0],imGIS.shape[1]))
for i in range(imGIS.shape[0]):
    for j in range(imGIS.shape[1]):
        if imGIS[i,j] == 0:
            iG[i,j]=0
        else:
            iG[i,j] = (clf.predict(im[i,j].reshape(-1,len(im[i,j]))))

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
plt.axis('off')
plt.pcolor(iG, cmap='jet')
plt.savefig(os.path.join('result', 'decode_map'+NO_data+'.png'), format='png')
plt.close()
print('decode map get finished')

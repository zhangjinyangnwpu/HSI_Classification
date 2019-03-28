import os
import scipy.io as sio
import numpy as np
times = 10
class_num = 9
path = 'result'
oa_list = list()
kappa_list = list()
aa = np.zeros([class_num,times])
for i in range(times):
    info_path = os.path.join(path,str(i),'result.mat')
    info = sio.loadmat(info_path)
    matrix = info['matrix']
    oa = info['oa']
    kappa = info['kappa']
    oa_list.append(oa)
    kappa_list.append(kappa)
    aa[:,i] = info['ac_list']

for i,value in enumerate(aa):
    print('%.2f'%(np.mean(value)*100))
print('\n')
for i,value in enumerate(aa):
    print('%.4f'%(np.var(value)*100))
print('\n')
print('oa:%.4f %.4f'%(np.mean(oa_list)*100,np.var(oa_list)*100))
print('kappa: %.2f %.4f'%(np.mean(kappa_list)*100,np.var(kappa_list)*100))





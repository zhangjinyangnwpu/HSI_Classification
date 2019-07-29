import os
import scipy.io as sio
import numpy as np
times = 5
class_num = 9
path = 'result'
oa_list = list()
aa_list = list()
kappa_list = list()
aa = np.zeros([class_num,times])
psnr_list = list()
for i in range(times):
    info_path = os.path.join(path,str(i),'result.mat')
    info = sio.loadmat(info_path)
    matrix = info['matrix']
    oa = info['oa']
    aa_t = info['aa']
    kappa = info['kappa']
    oa_list.append(oa)
    aa_list.append(aa_t)
    kappa_list.append(kappa)
    aa[:,i] = info['ac_list']
    psnr = sio.loadmat(os.path.join(path, str(i), 'decode_image.mat'))['psnr']
    psnr_list.append(psnr)

for i,value in enumerate(aa):
    print('%.2f'%(np.mean(value)*100))
print('\n')
for i,value in enumerate(aa):
    print('%.4f'%(np.var(value)*100))
print('\n')
# print("PSNR:", np.mean(psnr_list))
print(psnr_list)
print('oa:%.4f %.4f'%(np.mean(oa_list)*100,np.var(oa_list)*100))
print('aa:%.4f %.4f'%(np.mean(aa_list)*100,np.var(aa_list)*100))
print('kappa: %.2f %.4f'%(np.mean(kappa_list)*100,np.var(kappa_list)*100))
print('psnr:%.4f %.4f'%(np.mean(psnr_list),np.var(psnr_list)))





import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt
path_loss = os.path.join('result','0','loss_list.mat')
data_loss = sio.loadmat(path_loss)
total_loss = data_loss['loss_t'][0]
mse_loss = data_loss['mse'][0]
ce_loss = data_loss['ce'][0]
for id in range(5):
    print('--------------',id)
    plt.plot(total_loss)
    plt.savefig(os.path.join('result',str(id),'total_loss.png'))
    plt.close()
    plt.plot(mse_loss)
    plt.savefig(os.path.join('result',str(id),'mse_loss.png'))
    plt.close()
    plt.plot(ce_loss)
    plt.savefig(os.path.join('result',str(id),'ce_loss.png'))
    plt.close()

    path_loss = os.path.join('result',str(id),'result_list.mat')
    data_loss = sio.loadmat(path_loss)
    oa = data_loss['oa'][0]
    aa = data_loss['aa'][0]
    kappa = data_loss['kappa'][0]

    plt.plot(oa)
    plt.savefig(os.path.join('result',str(id),'oa.png'))
    plt.close()
    plt.plot(aa)
    plt.savefig(os.path.join('result',str(id),'aa.png'))
    plt.close()
    plt.plot(kappa)
    plt.savefig(os.path.join('result',str(id),'kappa.png'))
    plt.close()
    print('max oa',np.max(oa))
    print('max aa',np.max(aa))
    print('max kappa',np.max(kappa))

    path_loss = os.path.join('result',str(id),'psnr_list.mat')
    data_loss = sio.loadmat(path_loss)
    psnr = data_loss['psnr'][0]
    plt.plot(psnr)
    plt.savefig(os.path.join('result',str(id),'psnr.png'))
    plt.close()
    print('max psnr',np.max(psnr))
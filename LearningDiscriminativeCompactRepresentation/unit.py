import numpy as np
import scipy.io as sio
def PSNR(X,Y):
    mse = np.mean((X-Y)**2)
    if mse == 0:
        mse = 1e-10
    return 20*np.log10(1/np.sqrt(mse))

def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

# data = sio.loadmat('../../Data/PaviaU.mat')['paviaU']
# print(max_min(data))
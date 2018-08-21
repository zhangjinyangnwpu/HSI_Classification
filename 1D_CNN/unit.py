import numpy as np

def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

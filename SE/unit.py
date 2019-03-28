import numpy as np
import tensorflow as tf
def max_min(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def se_operation(feature):
    print('se operation',feature)
    mean_info = tf.reduce_mean(feature,[1,2,3])
    mean_info = tf.layers.flatten(mean_info)
    shape = mean_info.get_shape().as_list()[-1]
    mean_info = tf.layers.dense(mean_info,shape//4,activation=tf.nn.relu)
    mean_info = tf.layers.dense(mean_info,shape,activation=tf.nn.sigmoid)
    channel_info = mean_info
    mean_info = expand_dims(mean_info)
    se_feature = tf.multiply(feature,mean_info)
    return se_feature,channel_info

def expand_dims(x):
    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 1)
    x = tf.expand_dims(x, 1)
    return x

def skn_operation(feature):
    print('skn_operation',feature)
    shape = feature.get_shape().as_list()[-1]
    feature1 = tf.layers.conv3d(feature,shape,(1,1,3),(1,1,1),padding='same')
    feature2 = tf.layers.conv3d(feature,shape,(1,1,5),(1,1,1),padding='same')
    f12 = tf.add(feature1,feature2)
    _,fse1 = se_operation(f12)
    fse2 = tf.zeros_like(fse1)-fse1
    fse1 = expand_dims(fse1)
    fse2 = expand_dims(fse2)
    skn_feature = tf.add(tf.multiply(feature1,fse1),tf.multiply(feature2,fse2))
    return skn_feature






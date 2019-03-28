import tensorflow as tf
import numpy as np
from collections import Counter
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pathlib
import math
class Model():

    def __init__(self,args,sess):
        self.sess = sess
        self.data_name = args.data_name
        self.result = args.result
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        self.shape = info['shape']
        self.dim = info['dim']
        self.class_num = int(info['class_num'])
        self.data_gt = info['data_gt']
        self.log = args.log
        self.model = args.model
        self.cube_size = args.cube_size
        self.data_path = args.data_path
        self.iter_num = args.iter_num
        self.tfrecords = args.tfrecords
        self.best_oa = 0
        self.global_step = tf.Variable(0,trainable=False)
        self.training = tf.placeholder(bool)
        self.train_feed = False
        self.test_feed = False
        if args.use_lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rete,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr

        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.cube_size,self.cube_size,self.dim))
        self.label = tf.placeholder(dtype=tf.int64, shape=(None, 1))
        self.classifer = self.classifer

        self.pre_label = self.classifer(self.image,self.training)
        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.log),graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=100)

    def loss(self):
        with tf.variable_scope('loss'):
            loss_cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.label,self.pre_label,scope='loss_cross_entropy')
            loss_cross_entropy = tf.reduce_mean(loss_cross_entropy)
            self.loss_total = loss_cross_entropy
            tf.summary.scalar('loss_total',self.loss_total)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
        self.merged = tf.summary.merge_all()


    def classifer(self,feature,training=False):
        f_num = 64
        feature = tf.expand_dims(feature,4)
        print(feature)
        with tf.variable_scope('classifer',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(feature,f_num,(self.cube_size,1,8),strides=(1,1,3),padding='valid')
                conv0 = tf.layers.batch_normalization(conv0,training=training)
                conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0,f_num*2,(1,self.cube_size,3),strides=(1,1,2),padding='valid')
                conv1 = tf.layers.batch_normalization(conv1,training=training)
                conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(conv1,f_num*4,(1,1,3),strides=(1,1,2),padding='valid')
                conv2 = tf.layers.batch_normalization(conv2,training=training)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
                shape = int(conv2.get_shape().as_list()[3])
            with tf.variable_scope('global_info'):
                feature = tf.layers.conv3d(conv2,f_num*8,(1,1,shape),(1,1,1))
                feature = tf.layers.flatten(feature)
                print(feature)
            with tf.variable_scope('fc'):
                fc = tf.layers.dense(feature,512)
                fc = tf.layers.batch_normalization(fc)
                fc = tf.nn.relu(fc)
            with tf.variable_scope('logits'):
                logtic = tf.layers.dense(fc,self.class_num)
        return logtic


    def load(self, checkpoint_dir):
        print("Loading model ...")
        model_name = os.path.join(checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(model_name)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(model_name, ckpt_name))
            print("Load successful.")
            return True
        else:
            print("Load fail!!!")
            exit(0)

    def train(self,dataset):
        train_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'train_data.tfrecords'), type='train')
        init = tf.global_variables_initializer()
        self.best_oa = 0
        self.sess.run(init)
        for i in range(self.iter_num):
            train_data,train_label = self.sess.run(train_dataset)
            # print(train_data.shape,train_label.shape)
            l,_,summery,lr= self.sess.run([self.loss_total,self.optimizer,self.merged,self.lr],feed_dict={self.image:train_data,self.label:train_label,self.training:self.train_feed})
            if i % 1000 == 0:
                print(i,'step:',l,'learning rate:',lr)
            if i % 10000 == 0 and i!=0:
                self.saver.save(self.sess,os.path.join(self.model,self.model_name),global_step=i)
                print('saved...')
                self.test(dataset)
                self.save_decode_map(dataset)
            self.summary_write.add_summary(summery,i)

    def test(self,dataset):
        test_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.int64)
        try:
            while True:
                test_data, test_label = self.sess.run(test_dataset)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:test_data,self.label:test_label,self.training:self.test_feed})
                pre_label = np.argmax(pre_label,1)
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label[i]]+=1
        except tf.errors.OutOfRangeError:
            print("test end!")

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
        if self.best_oa < oa:
            self.best_oa = oa
        sio.savemat(os.path.join(self.result, 'result'+self.data_name+'.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix,'best_oa':self.best_oa})

    def save_decode_map(self,dataset):
        map_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'map_data.tfrecords'), type='map')
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        data_gt = info['data_gt']
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.pcolor(data_gt, cmap='jet')
        plt.savefig(os.path.join(self.result, 'groundtrouth.png'), format='png')
        plt.close()
        print('Groundtruth map get finished')
        de_map = np.zeros(data_gt.shape,dtype=np.int32)
        try:
            while True:
                map_data,pos = self.sess.run(map_dataset)
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:map_data,self.training:self.test_feed})
                pre_label = np.argmax(pre_label,1)
                for i in range(pre_label.shape[0]):
                    [r,c]=pos[i]
                    de_map[r,c] = pre_label[i] + 1
        except tf.errors.OutOfRangeError:
            print("test end!")
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
        plt.axis('off')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map'+self.data_name+'.png'), format='png')
        plt.close()
        print('decode map get finished')
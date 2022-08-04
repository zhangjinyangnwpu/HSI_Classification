import tensorflow as tf
import numpy as np
from collections import Counter
import scipy.io as sio
import os
import matplotlib.pyplot as plt
import pathlib
import math
import unit
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
        self.train_feed = True
        self.test_feed = False
        if args.use_lr_decay:
            self.lr = tf.train.exponential_decay(learning_rate=args.lr,
                                             global_step=self.global_step,
                                             decay_rate=args.decay_rete,
                                             decay_steps=args.decay_steps)
        else:
            self.lr = args.lr
        self.image = tf.placeholder(dtype=tf.float32, shape=(None, self.cube_size,self.cube_size,self.dim))
        self.label = tf.placeholder(dtype=tf.float32, shape=(None, self.class_num))
        self.classifer = unit.classifer_share1d
        self.pre_label = self.classifer(self.image,self.class_num,self.cube_size,self.training,reuse=tf.AUTO_REUSE)
        self.model_name = os.path.join('model.ckpt')
        self.loss()
        self.summary_write = tf.summary.FileWriter(os.path.join(self.log),graph=self.sess.graph)
        self.saver = tf.train.Saver(max_to_keep=5)
    
    def loss(self):
        with tf.variable_scope('loss'):
            loss = tf.losses.softmax_cross_entropy(self.label,self.pre_label,scope='loss_cross_entropy')
            loss_cross_entropy = tf.reduce_mean(loss)
            self.loss_total = loss_cross_entropy
            tf.summary.scalar('loss_total',self.loss_total)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss_total,global_step=self.global_step)
            self.optimizer = tf.train.MomentumOptimizer(self.lr,momentum=0.9).minimize(self.loss_total, global_step=self.global_step)
        self.merged = tf.summary.merge_all()

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
            train_label = train_label.reshape(-1)
            train_label = np.eye(self.class_num)[train_label]
            l,_,summery,lr= self.sess.run([self.loss_total,self.optimizer,self.merged,self.lr],feed_dict={self.image:train_data,self.label:train_label,self.training:self.train_feed})
            if i % 100 == 0:
                print(i,'step:',l,'learning rate:',lr)
            if i % 1000 == 0 and i!=0:
                self.saver.save(self.sess,os.path.join(self.model,self.model_name),global_step=i)
                print('saved...')
                self.test(dataset)
            self.summary_write.add_summary(summery,i)

    def test(self,dataset):
        test_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'test_data.tfrecords'), type='test')
        acc_num,test_num = 0,0
        matrix = np.zeros((self.class_num,self.class_num),dtype=np.int64)
        try:
            while True:
                test_data, test_label = self.sess.run(test_dataset)
                test_label_new = test_label[:]
                test_label = test_label.reshape(-1)
                test_label = np.eye(self.class_num)[test_label]
                pre_label = self.sess.run(self.pre_label, feed_dict={self.image:test_data,self.label:test_label,self.training:self.test_feed})
                pre_label = np.argmax(pre_label,1)
                pre_label = np.expand_dims(pre_label,1)
                acc_num += np.sum((pre_label==test_label_new))
                test_num += test_label.shape[0]
                print(acc_num,test_num,acc_num/test_num)
                for i in range(pre_label.shape[0]):
                    matrix[pre_label[i],test_label_new[i]]+=1
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
        sio.savemat(os.path.join(self.result, 'result.mat'), {'oa': oa,'aa':aa,'kappa':kappa,'ac_list':ac_list,'matrix':matrix,'best_oa':self.best_oa})

    def save_decode_map(self,dataset):
        map_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'map_data.tfrecords'), type='map')
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        data_gt = info['data_gt'][::-1]
        fig, _ = plt.subplots()
        height, width = data_gt.shape
        fig.set_size_inches(width/100.0, height/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        # plt.margins(0,0)
        plt.axis('off')
        plt.axis('equal')
        plt.pcolor(data_gt, cmap='jet')
        plt.savefig(os.path.join(self.result, 'groundtrouth_'+self.data_name+'.png'), \
            format='png',dpi=800)
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
        de_map = de_map[::-1]
        fig, _ = plt.subplots()
        height, width = de_map.shape
        fig.set_size_inches(width/100.0, height/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.axis('off')
        plt.axis('equal')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map_'+self.data_name+'.png'),\
             format='png',dpi=800)#bbox_inches='tight',pad_inches=0)
        plt.close()
        print('decode map get finished')
    
    def save_decode_seg_map(self,dataset):
        map_dataset = dataset.data_parse(os.path.join(self.tfrecords, 'map_data_seg.tfrecords'), type='map_seg')
        info = sio.loadmat(os.path.join(self.result,'info.mat'))
        data_gt = info['data_gt']
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
        de_map = de_map[::-1]
        fig, _ = plt.subplots()
        height, width = de_map.shape
        fig.set_size_inches(width/100.0, height/100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
        plt.axis('off')
        plt.axis('equal')
        plt.pcolor(de_map, cmap='jet')
        plt.savefig(os.path.join(self.result, 'decode_map_seg'+self.data_name+'.png'),\
             format='png',dpi=600)#bbox_inches='tight',pad_inches=0)
        plt.close()
        print('seg decode map get finished')
import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import random
import time
import matplotlib.pyplot as plt

class NC(object):
    def __init__(self,sess,args,id=0):

        self.sess = sess
        self.id = str(id)
        self.train_path = args.data_path
        self.result = args.result
        self.checkpoint = args.checkpoint
        self.log = args.log
        self.TFrecords = args.TFrecords
        self.iterate_num = args.iterate_num

        if not os.path.exists(self.result):
            os.mkdir(self.result)
        if not os.path.exists(self.TFrecords):
            os.mkdir(self.TFrecords)
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        if not os.path.exists(self.log):
            os.mkdir(self.log)

        self.batch_size = args.batch_size
        self.test_batch = args.test_batch
        self.cube_size = args.cube_size
        self.train_num = args.train_num
        self.dataset = args.dataset
        self.iterate_num = args.iterate_num
        self.reuse = args.reuse
        self.global_step = tf.Variable(initial_value=tf.constant(0),trainable=True)
        self.learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                             global_step=self.global_step,
                                             decay_rate=0.9,
                                             decay_steps=5000)

        if self.dataset == 'PaviaU':
            self.dim = 103
            self.num_class = 9
        elif self.dataset == 'Indian_pines':
            self.dim = 220
            self.num_class = 9
        elif self.dataset == 'Salinas':
            self.dim = 204
            self.num_class = 16
        else:
            print("dataset don't set!")
            exit(0)

        self.cube = tf.placeholder(
            tf.float32, shape=(None, self.cube_size, self.cube_size, self.dim,1))
        self.vector = tf.placeholder(
            tf.float32,shape=(None,self.dim)
        )
        self.labels = tf.placeholder(tf.int32, shape=(None, 1))
        self.classifer = self.classifer_3d_cnn_layer5

    def classifer_3d_cnn_layer5(self,cube,vector):
        dim = 16
        print(cube)
        init_kernel = tf.truncated_normal_initializer(0.001)
        ks = [[3,3,3],[3,1,3],[1,3,3],[1,1,3]]
        s = [[1,1,2],[1,1,2],[1,1,2],[1,1,2]]
        with tf.variable_scope('network',reuse=tf.AUTO_REUSE):
            with tf.variable_scope('conv0'):
                conv0 = tf.layers.conv3d(cube,dim,ks[0],s[0],padding='same', kernel_initializer=init_kernel)
                conv0 = tf.layers.batch_normalization(conv0,momentum=0.9,epsilon=1e-4)
                conv0 = tf.nn.relu(conv0)
                print(conv0)
            with tf.variable_scope('conv1'):
                conv1 = tf.layers.conv3d(conv0,dim*2,ks[1],s[1], kernel_initializer=init_kernel)
                conv1 = tf.layers.batch_normalization(conv1,momentum=0.9,epsilon=1e-4)
                conv1 = tf.nn.relu(conv1)
                print(conv1)
            with tf.variable_scope('conv2'):
                conv2 = tf.layers.conv3d(conv1,dim*4,ks[2],s[2],kernel_initializer=init_kernel)
                conv2 = tf.layers.batch_normalization(conv2,momentum=0.9,epsilon=1e-4)
                conv2 = tf.nn.relu(conv2)
                print(conv2)
            with tf.variable_scope('conv3'):
                conv3 = tf.layers.conv3d(conv2,dim*8,ks[3],s[3],padding='same', kernel_initializer=init_kernel)
                conv3 = tf.layers.batch_normalization(conv3,momentum=0.9,epsilon=1e-4)
                conv3 = tf.nn.relu(conv3)
                print(conv3)
            with tf.variable_scope('score'):
                shapt_t = int(conv3.get_shape().as_list()[3])
                score = tf.layers.conv3d(conv3,self.num_class,[1,1,shapt_t], [1,1,1])
                print(score)
            score = tf.layers.flatten(score)
            print(score)
        return score

    def data_prepare(self):
        if self.dataset == 'Salinas':
            data = sio.loadmat(os.path.join(self.train_path, 'Salinas.mat'))
            data_gt = sio.loadmat(os.path.join(self.train_path, 'Salinas_gt.mat'))
            im = data['salinas_corrected']
            imGIS = data_gt['salinas_gt']
        if self.dataset == 'PaviaU':
            data = sio.loadmat(os.path.join(self.train_path, 'PaviaU.mat'))
            data_gt = sio.loadmat(os.path.join(self.train_path, 'PaviaU_gt.mat'))
            im = data['paviaU']
            imGIS = data_gt['paviaU_gt']
        if self.dataset == 'Indian_pines':
            data = sio.loadmat(os.path.join(self.train_path, 'Indian_pines.mat'))
            data_gt = sio.loadmat(os.path.join(self.train_path, 'Indian_pines_gt.mat'))
            im = data['indian_pines']
            imGIS = data_gt['indian_pines_gt']
            origin_num = np.zeros(shape=[17], dtype=int)
            for i in range(imGIS.shape[0]):
                for j in range(imGIS.shape[1]):
                    for k in range(1, 17):
                        if imGIS[i][j] == k:
                            origin_num[k] += 1
            index = 0
            data_num = np.zeros(shape=[9], dtype=int)  # per calsses's num
            data_label = np.zeros(shape=[9], dtype=int)  # original labels
            for i in range(len(origin_num)):
                if origin_num[i] > 400:
                    data_num[index] = origin_num[i]
                    data_label[index] = i
                    index += 1
            iG = np.zeros([imGIS.shape[0], imGIS.shape[1]], dtype=imGIS.dtype)
            for i in range(imGIS.shape[0]):
                for j in range(imGIS.shape[1]):
                    if imGIS[i, j] in data_label:
                        for k in range(len(data_label)):
                            if imGIS[i][j] == data_label[k]:
                                iG[i, j] = k + 1
                                continue
            imGIS = iG

        plt.pcolor(imGIS, cmap='jet')
        result_path = os.path.join(self.result, self.dataset)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        result_path = os.path.join(result_path, self.id)
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        plt.savefig(os.path.join(result_path, 'groundtrouth.png'), format='png')
        plt.close()
        self.num_class = np.max(imGIS)

        im = (im - float(np.min(im)))
        im = im / np.max(im)

        lable_pos = {}  # per class's pos
        for i in range(1, self.num_class + 1):
            lable_pos[i] = []
        for row in range(imGIS.shape[0]):
            for col in range(imGIS.shape[1]):
                for t in range(1, self.num_class + 1):
                    if imGIS[row, col] == 0: continue
                    if imGIS[row, col] == t:
                        lable_pos[t].append([row, col])
                        continue
        t = time.time()
        random.seed(t)

        if not os.path.exists(os.path.join(self.result,self.dataset,self.id)):
            os.mkdir(os.path.join(self.result,self.dataset,self.id))

        f = open(os.path.join(self.result,self.dataset,self.id,'seed_pos.txt'),'w')
        f.write('seed:'+str(t)+'\n')
        for i in range(1, self.num_class + 1):
            train_indices = random.sample(lable_pos[i], self.train_num)
            f.write(str(train_indices))
            for k in range(len(train_indices)):
                imGIS[train_indices[k][0], train_indices[k][1]] = i + self.num_class + 1
        f.close()

        self.shape = im.shape
        self.dim = im.shape[2]
        trainclass = {}
        testclass = {}
        for i in range(1, self.num_class + 1):
            trainclass[i] = []
            testclass[i] = []
        train_map = np.zeros([imGIS.shape[0], imGIS.shape[1]], dtype=imGIS.dtype)
        for i in range(imGIS.shape[0]):
            for j in range(imGIS.shape[1]):
                if imGIS[i, j] > 0 and imGIS[i, j] <= self.num_class:
                    testclass[imGIS[i, j]].append([i,j])
                elif imGIS[i, j] > self.num_class + 1 and imGIS[i, j] <= self.num_class * 2 + 1:
                    trainclass[imGIS[i, j] - self.num_class - 1].append([i,j])
                    train_map[i, j] = imGIS[i, j] -self.num_class-1
        sio.savemat(os.path.join(self.result, self.dataset,self.id, 'train_map.mat'), {'train_map': train_map})

        def neighbor_add(row, col, labels, w_size=3, flag=True):  # 给出 row，col和标签，返回w_size大小的cube，flag=True表示为训练样本
            t = w_size // 2
            cube = np.zeros(shape=[w_size, w_size, im.shape[2]])
            for i in range(-t, t + 1):
                for j in range(-t, t + 1):
                    if i + row < 0 or i + row >= im.shape[0] or j + col < 0 or j + col >= im.shape[1]:
                        # s = random.sample(trainclass[j],3)
                        if flag == True:
                            s = random.sample(trainclass[labels], 1)
                            cube[i + t, j + t] = s[0][1]
                        else:
                            s = random.sample(testclass[labels], 1)
                            cube[i + t, j + t] = s[0][1]
                    else:
                        cube[i + t, j + t] = im[i + row, j + col]
            return cube

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        filename = os.path.join(self.TFrecords,
                                self.dataset +'_train_'+ self.id + '_cube_' + str(self.cube_size) +
                                '_train_num_' + str(self.train_num) + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename)

        index = 0
        for i in range(1, self.num_class + 1):
            n = len(trainclass[i])
            for j in range(n):
                [row, col] = trainclass[i][j]
                # print(row, col)
                cube = neighbor_add(row, col, i, w_size=self.cube_size, flag=True)
                vector = im[row, col,:]
                label = i
                cube = cube.tostring()
                vector = vector.tostring()
                example = tf.train.Example(features=(tf.train.Features(feature={
                    'cube': _bytes_feature(cube),
                    'vector': _bytes_feature(vector),
                    'label': _int64_feature(label)
                })))
                index += 1
                # print("pre for train %d"%index)
                writer.write(example.SerializeToString())
        self.total_train_num = index
        writer.close()

        filename_test = os.path.join(self.TFrecords,
                                self.dataset +'_test_'+ self.id+'_cube_' + str(self.cube_size) + '.tfrecords')
        writer = tf.python_io.TFRecordWriter(filename_test)

        index = 0
        for i in range(1, self.num_class + 1):
            n = len(testclass[i])
            for j in range(n):
                [row, col] = testclass[i][j]
                cube = neighbor_add(row, col, i, w_size=self.cube_size, flag=True)
                vector = im[row, col,:]
                label = i
                cube = cube.tostring()
                vector = vector.tostring()
                example = tf.train.Example(features=(tf.train.Features(feature={
                    'cube': _bytes_feature(cube),
                    'vector': _bytes_feature(vector),
                    'label': _int64_feature(label)
                })))
                index += 1
                # print("pre for train %d"%index)
                writer.write(example.SerializeToString())
        self.total_test_num = index
        writer.close()

    def load(self, checkpoint_dir):
        print("Loading model ...")
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False


    def dataset_input(self,filename,type):
        dataset = tf.data.TFRecordDataset([filename])

        def parser1(record):
            keys_to_features = {
                'cube': tf.FixedLenFeature([], tf.string),
                'vector': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            cube = tf.decode_raw(features['cube'], tf.float64)
            vector = tf.decode_raw(features['vector'], tf.float64)
            train_label = tf.cast(features['label'], tf.int32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            cube = tf.reshape(cube, shape)
            vector = tf.reshape(vector, [self.dim])
            train_label = tf.reshape(train_label, [1])
            return cube,vector, train_label

        def parser2(record):
            keys_to_features = {
                'cube': tf.FixedLenFeature([], tf.string),
                'vector': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            cube = tf.decode_raw(features['cube'], tf.float64)
            vector = tf.decode_raw(features['vector'], tf.float64)
            test_label = tf.cast(features['label'], tf.int32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            cube = tf.reshape(cube, shape)
            vector = tf.reshape(vector, [self.dim])
            test_label = tf.reshape(test_label, [1])
            return cube, vector, test_label

        if type == 0:
            dataset = dataset.map(parser1,num_parallel_calls=24)
            dataset = dataset.shuffle(buffer_size=10000)

            dataset = dataset.repeat()
            dataset = dataset.batch(self.batch_size)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            return features
        if type ==1:
            dataset = dataset.map(parser2)
            dataset = dataset.batch(self.test_batch)
            iterator = dataset.make_one_shot_iterator()
            features = iterator.get_next()
            return features

    def train(self):
        with tf.variable_scope('inference'):
            logits = self.classifer(self.cube, self.vector)
        labels = tf.reshape(self.labels, [self.batch_size])
        labels = tf.one_hot(labels-1,self.num_class)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
        loss  = tf.reduce_mean(cross_entropy)
        variable = tf.trainable_variables()
        loss += tf.add_n([tf.nn.l2_loss(v) for v in variable if 'kernel' in v.name])*1e-3
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss,global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        filename = os.path.join(self.TFrecords,
                                self.dataset + '_train_' + self.id + '_cube_' + str(self.cube_size) +
                                '_train_num_' + str(self.train_num) + '.tfrecords')
        dataset = self.dataset_input(filename, 0)
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if self.reuse:
            if self.load(self.checkpoint):
                print('load successful...')
            else:
                print('load fail!!!')
                return

        model_name = os.path.join(self.checkpoint, self.dataset+'3D_PCA.model')
        print('starting...\n')
        iterate_num = self.iterate_num
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess,coord=coord)
        for step in range(iterate_num):
            cube, _, train_label = self.sess.run(dataset)
            self.sess.run([train_step],feed_dict={self.cube:cube,self.labels:train_label})
            if step%10000 == 0 and step!=0:
                self.saver.save(self.sess,model_name,step)
                print('step %d saved'%step)
                self.test()
            if step%100 == 0 and step!=0:
                l,lr = self.sess.run([loss,self.learning_rate],feed_dict={self.cube:cube,self.labels:train_label})
                print(step,':saved','loss:',l,'learning rate:',lr)
        coord.request_stop()
        coord.join(threads)

    def test(self):
        with tf.variable_scope('inference'):
            logits = self.classifer(self.cube, self.vector)
        y_conv = tf.nn.softmax(logits)
        y_ = tf.argmax(y_conv,1)
        conf_matrix = tf.confusion_matrix(self.labels-1,y_,num_classes=self.num_class)

        if self.load(self.checkpoint):
            print('load successful...')
        else:
            print('load fail!!!')
            return
        filename_test = os.path.join(self.TFrecords,
                                     self.dataset + '_test_' + self.id + '_cube_' + str(self.cube_size) + '.tfrecords')
        dataset = self.dataset_input(filename_test, 1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        matrix = np.zeros((self.num_class,self.num_class),dtype=np.int32)
        try:
            while True:
                cube, _, test_label = self.sess.run(dataset)
                cm,ll = self.sess.run([conf_matrix,y_], feed_dict={self.cube: cube,self.labels:test_label})
                matrix += cm
        except tf.errors.OutOfRangeError:
            print("end!")

        coord.request_stop()
        coord.join(threads)
        print(matrix)
        ac_list = []
        for i in range(len(matrix)):
            ac = matrix[i, i] / sum(matrix[i,:])
            ac_list.append(ac)
            print('(',matrix[i, i],'/',sum(matrix[i,:]),')',ac)
        print(np.sum(np.trace(matrix)),'/',np.sum(matrix))
        accuracy = np.sum(np.trace(matrix)) / np.sum(matrix)
        print('accuracy:',accuracy)
        # 计算kappa值
        kk = 0
        for i in range(matrix.shape[0]):
            kk += np.sum(matrix[i]) * np.sum(matrix[:, i])
        pe = kk / (np.sum(matrix) * np.sum(matrix))
        pa = np.trace(matrix) / np.sum(matrix)
        kappa = (pa - pe) / (1 - pe)
        print('kappa:',kappa)
        sio.savemat(os.path.join(self.result,self.dataset,self.id,'matrix_kappa_'+self.id+'.mat'),{
            'matrix':matrix,
            'kappa':kappa,
            'accuracy':accuracy,
            'ac_list':ac_list
        })

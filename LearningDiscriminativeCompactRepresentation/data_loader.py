import pathlib,random
import scipy.io as sio
import numpy as np
import tensorflow as tf
import unit,os

class Data():

    def __init__(self,args):
        self.args = args
        self.data_path = args.data_path
        self.train_num = args.train_num
        self.fix_seed = args.fix_seed
        self.seed = args.seed
        self.tfrecords = args.tfrecords
        self.cube_size = args.cube
        self.data_name = args.data_name

        self.data_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_name + '.mat')))
        self.data_gt_dict = sio.loadmat(str(pathlib.Path(self.data_path, self.data_name + '_gt.mat')))
        data_name = [t for t in list(self.data_dict.keys()) if not t.startswith('__')][0]
        data_gt_name = [t for t in list(self.data_gt_dict.keys()) if not t.startswith('__')][0]
        self.data = self.data_dict[data_name]
        self.data = unit.max_min(self.data).astype(np.float32)
        self.data_gt = self.data_gt_dict[data_gt_name].astype(np.int64)
        self.dim = self.data.shape[2]

    def neighbor_add(self, row, col, labels=1, w_size=3, flag='train'):  # row，col,w_size大小的cube，flag=True
        t = w_size // 2
        cube = np.zeros(shape=[w_size, w_size, self.data.shape[2]])
        for i in range(-t, t + 1):
            for j in range(-t, t + 1):
                if i + row < 0 or i + row >= self.data.shape[0] or j + col < 0 or j + col >= self.data.shape[1]:
                    if flag == 'train':
                        s = random.sample(self.train_pos[labels], 1)
                        cube[i + t, j + t] = s[0][1]
                    elif flag == 'test':
                        s = random.sample(self.test_pos[labels], 1)
                        cube[i + t, j + t] = s[0][1]
                    elif flag == 'sae':
                        cube[i + t, j + t] = self.data[row,col]
                else:
                    cube[i + t, j + t] = self.data[i + row, j + col]
        return cube

    def read_data(self):
        data = self.data
        data_gt = self.data_gt

        if self.data_name == 'Indian_pines':
            imGIS = data_gt

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

            data_gt = imGIS
            self.data_gt = data_gt

        class_num = np.max(data_gt)
        data_pos = {i:[] for i in range(1,class_num+1)}
        for i in range(data_gt.shape[0]):
            for j in range(data_gt.shape[1]):
                for k in range(1,class_num+1):
                    if data_gt[i,j]==k:
                        data_pos[k].append([i,j])
                        continue

        if self.fix_seed:
            random.seed(self.seed)
        train_pos = dict()
        test_pos = dict()
        for k,v in data_pos.items():
            if self.train_num > 0 and self.train_num <1:
                train_num = int(self.train_num*len(v))
            else:
                train_num = self.train_num
            train_pos[k] = random.sample(v,train_num)
            test_pos[k] = [i for i in v if i not in train_pos[k]]
        self.train_pos = train_pos
        self.test_pos = test_pos

        train_pos_all = list()
        test_pos_all = list()
        for k,v in train_pos.items():
            for t in v:
                train_pos_all.append([k,t])
        for k,v in test_pos.items():
            for t in v:
                test_pos_all.append([k,t])

        self.train_len = len(train_pos_all)
        sio.savemat(os.path.join(self.args.tfrecords,'info.mat'),{'total_train_num':self.train_len,
                                                                  'shape':self.data.shape,
                                                                  'dim':self.data.shape[2],
                                                                  'class_num':np.max(self.data_gt),
                                                                  'data':self.data,
                                                                  'data_gt':self.data_gt})

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        # train data
        train_data_name = os.path.join(self.tfrecords,'train_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(train_data_name)
        for i in train_pos_all:
            [r,c] = i[1]
            cube = self.neighbor_add(r,c,i[0],w_size=self.cube_size,flag='train')
            cube = cube.astype(np.float32).tostring()
            label = i[0] - 1
            label = np.array(np.array(label).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'traindata':_bytes_feature(cube),
                    'trainlabel':_int64_feature(label)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()
        #test data
        test_data_name = os.path.join(self.tfrecords, 'test_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(test_data_name)
        for i in test_pos_all:
            [r, c] = i[1]
            cube = self.neighbor_add(r,c,i[0],w_size=self.cube_size,flag='test')
            cube = cube.astype(np.float32).tostring()
            label = i[0] - 1
            label = np.array(np.array(label).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'testdata': _bytes_feature(cube),
                    'testlabel': _int64_feature(label)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()

        pos_all = list()
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                pos_all.append([i,j])
        if self.fix_seed:
            random.seed(self.seed)
        sae_pos_all = random.sample(pos_all,int(self.args.sae_ratio*len(pos_all)))

        # sae train data
        sae_data_name = os.path.join(self.tfrecords, 'sae_train_data_'+str(self.args.sae_ratio)+'.tfrecords')
        writer = tf.python_io.TFRecordWriter(sae_data_name)

        for pos in sae_pos_all:
            i,j = pos
            cube = self.neighbor_add(i, j,w_size=self.cube_size ,flag='sae')
            cube = cube.astype(np.float32).tostring()
            row = np.array(np.array(i).astype(np.int64))
            col = np.array(np.array(j).astype(np.int64))
            example = tf.train.Example(features=(tf.train.Features(
                feature={
                    'sae_train': _bytes_feature(cube),
                    'row':_int64_feature(row),
                    'col':_int64_feature(col)
                }
            )))
            writer.write(example.SerializeToString())
        writer.close()

        # sae test data
        sae_data_name = os.path.join(self.tfrecords, 'sae_test_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(sae_data_name)
        for i in range(self.data_gt.shape[0]):
            for j in range(self.data_gt.shape[1]):
                cube = self.neighbor_add(i, j, w_size=self.cube_size, flag='sae')
                cube = cube.astype(np.float32).tostring()
                row = np.array(np.array(i).astype(np.int64))
                col = np.array(np.array(j).astype(np.int64))
                example = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'sae_test': _bytes_feature(cube),
                        'row': _int64_feature(row),
                        'col': _int64_feature(col)
                    }
                )))
                writer.write(example.SerializeToString())
        writer.close()

        # demap data
        train_data_name = os.path.join(self.tfrecords, 'demap_data.tfrecords')
        writer = tf.python_io.TFRecordWriter(train_data_name)
        for k,v in data_pos.items():
            for i in v:
                [r, c] = i
                cube = self.neighbor_add(r, c, self.data_gt[r,c], w_size=self.cube_size, flag='sae')
                cube = cube.astype(np.float32).tostring()
                row = np.array(np.array(r).astype(np.int64))
                col = np.array(np.array(c).astype(np.int64))
                example = tf.train.Example(features=(tf.train.Features(
                    feature={
                        'demap': _bytes_feature(cube),
                        'row': _int64_feature(row),
                        'col': _int64_feature(col),
                    }
                )))
                writer.write(example.SerializeToString())
        writer.close()

    def data_parse(self,filename,type='train'):
        dataset = tf.data.TFRecordDataset([filename])
        def parser_train(record):
            keys_to_features = {
                'traindata': tf.FixedLenFeature([], tf.string),
                'trainlabel': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            train_data = tf.decode_raw(features['traindata'], tf.float32)
            train_label = tf.cast(features['trainlabel'], tf.int32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            train_data = tf.reshape(train_data, shape)
            train_label = tf.reshape(train_label, [1])
            return train_data, train_label
        def parser_test(record):
            keys_to_features = {
                'testdata': tf.FixedLenFeature([], tf.string),
                'testlabel': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            test_data = tf.decode_raw(features['testdata'], tf.float32)
            test_label = tf.cast(features['testlabel'], tf.int32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            test_data = tf.reshape(test_data, shape)
            test_label = tf.reshape(test_label, [1])
            return test_data, test_label
        def parser_sae_train(record):
            keys_to_features = {
                'sae_train': tf.FixedLenFeature([], tf.string),
                'row': tf.FixedLenFeature([], tf.int64),
                'col': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            sae_data = tf.decode_raw(features['sae_train'], tf.float32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            sae_data = tf.reshape(sae_data, shape)
            row = tf.cast(features['row'], tf.int32)
            col = tf.cast(features['col'], tf.int32)
            row = tf.reshape(row, [1])
            col = tf.reshape(col, [1])
            return row,col,sae_data
        def parser_sae_test(record):
            keys_to_features = {
                'sae_test': tf.FixedLenFeature([], tf.string),
                'row': tf.FixedLenFeature([], tf.int64),
                'col': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            sae_data = tf.decode_raw(features['sae_test'], tf.float32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            sae_data = tf.reshape(sae_data, shape)
            row = tf.cast(features['row'], tf.int32)
            col = tf.cast(features['col'], tf.int32)
            row = tf.reshape(row, [1])
            col = tf.reshape(col, [1])
            return row,col,sae_data
        def parser_demap(record):
            keys_to_features = {
                'demap': tf.FixedLenFeature([], tf.string),
                'row': tf.FixedLenFeature([], tf.int64),
                'col': tf.FixedLenFeature([], tf.int64),
            }
            features = tf.parse_single_example(record, features=keys_to_features)
            demap = tf.decode_raw(features['demap'], tf.float32)
            shape = [self.cube_size, self.cube_size, self.dim, 1]
            demap = tf.reshape(demap, shape)
            row = tf.cast(features['row'], tf.int32)
            col = tf.cast(features['col'], tf.int32)
            row = tf.reshape(row, [1])
            col = tf.reshape(col, [1])
            return row,col,demap

        if type == 'train':
            dataset = dataset.map(parser_train)
            dataset = dataset.shuffle(buffer_size=20000)
            dataset = dataset.batch(self.args.supervise_batch)
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'test':
            dataset = dataset.map(parser_test)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'sae_train':
            dataset = dataset.map(parser_sae_train)
            dataset = dataset.shuffle(buffer_size=20000)
            dataset = dataset.batch(self.args.batch_size-self.args.supervise_batch)
            dataset = dataset.repeat()
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'sae_test':
            dataset = dataset.map(parser_sae_test)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
        if type == 'demap':
            dataset = dataset.map(parser_demap)
            dataset = dataset.batch(self.args.test_batch)
            iterator = dataset.make_one_shot_iterator()
            return iterator.get_next()
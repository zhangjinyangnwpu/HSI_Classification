import argparse
import os
import tensorflow as tf

from data_loader import Data
from model import Model
from datetime import datetime
parser = argparse.ArgumentParser(description='Compresion and Classification for HSI')

parser.add_argument('--result', dest='result', default='result')# path result, will contain a sub floder with id
parser.add_argument('--log', dest='log', default='log')
parser.add_argument('--model', dest='model', default='model')# path to save the model
parser.add_argument('--tfrecords', dest='tfrecords', default='tfrecords')
parser.add_argument('--data_path', dest='data_path', default='../../Data')
parser.add_argument('--data_name', dest='data_name', default='PaviaU')
parser.add_argument('--fix_seed',dest='fix_seed',default=False)
parser.add_argument('--seed', dest='seed', default=666)


parser.add_argument('--learningrate', dest='lr', default=0.0001)
parser.add_argument('--use_lr_decay', dest='use_lr_decay', default=True)
parser.add_argument('--decay_rete', dest='decay_rete', default=0.96)
parser.add_argument('--decay_steps', dest='decay_steps', default=20000)
parser.add_argument('--weight_learnable', dest='weight_learnable', default=False)

parser.add_argument('--train_num', dest='train_num', default=200)  # intger for number and decimal for percentage
parser.add_argument('--batch_size', dest='batch_size', default=200)
parser.add_argument('--supervise_batch', dest='supervise_batch', default=150)# batch for classification
parser.add_argument('--test_batch', dest='test_batch', default=2000)
parser.add_argument('--cube', dest='cube', default=1)# 1 means use a pixel with vector
parser.add_argument('--compression_ratio', dest='c_r', default=10)  # compression ratio
parser.add_argument('--ratio_cc', dest='ratio_cc', default=100)  # mse and cross entorpy ratio
parser.add_argument('--sae_ratio', dest='sae_ratio', default=1) # the percentage of training number for compresion and decompresion

parser.add_argument('--iter_num', dest='iter_num', default=400001)
parser.add_argument('--get_decode_map', dest='get_decode_map', default=True)# get the classification map
parser.add_argument('--get_decode_image', dest='get_decode_image', default=True)# get the decode HSI,
parser.add_argument('--get_feature', dest='get_feature', default=True)# get the compresion HSI
parser.add_argument('--load_model', dest='load_model', default=False)# load the trained model or not

args = parser.parse_args()
if not os.path.exists(args.model):
    os.mkdir(args.model)
if not os.path.exists(args.log):
    os.mkdir(args.log)
if not os.path.exists(args.result):
    os.mkdir(args.result)
if not os.path.exists(args.tfrecords):
    os.mkdir(args.tfrecords)

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    time_list = []
    for i in range(5):# 5 runs for average
        args.id = str(i)# id
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            args.result = os.path.join(args.result, args.id)
            args.log = os.path.join(args.log, args.id)
            args.model = os.path.join(args.model, args.id)
            args.tfrecords = os.path.join(args.tfrecords, args.id)
            if not os.path.exists(args.model):
                os.mkdir(args.model)
            if not os.path.exists(args.log):
                os.mkdir(args.log)
            if not os.path.exists(args.result):
                os.mkdir(args.result)
            if not os.path.exists(args.tfrecords):
                os.mkdir(args.tfrecords)

            data_model = Data(args)
            data_model.read_data()
            dataset_train = data_model.data_parse(
                os.path.join(args.tfrecords, 'train_data.tfrecords'), type='train')
            dataset_test = data_model.data_parse(
                os.path.join(args.tfrecords, 'test_data.tfrecords'), type='test')
            dataset_sae_train = data_model.data_parse(
                os.path.join(args.tfrecords, 'sae_train_data_' + str(args.sae_ratio) + '.tfrecords'), type='sae_train')
            dataset_sae_test = data_model.data_parse(
                os.path.join(args.tfrecords, 'sae_test_data.tfrecords'), type='sae_test')
            dataset_de_map = data_model.data_parse(
                os.path.join(args.tfrecords, 'demap_data' + '.tfrecords'), type='demap')

            model = Model(args,sess)
            if args.load_model:
                model.load(args.model)
            else:
                model.train(dataset_train,dataset_sae_train,data_model)
            a = datetime.now()
            model.test(dataset_test)
            b = datetime.now()
            time_list.append((b-a).total_seconds())
            if args.get_decode_map:
                model.save_decode_map(dataset_de_map)
            if args.get_decode_image:
                model.get_decode_image(dataset_sae_test)
            if args.get_feature:
                model.get_feature(dataset_sae_test)
            args.result = 'result'
            args.log = 'log'
            args.tfrecords = 'tfrecords'
            args.model = 'model'

if __name__ == '__main__':
    main()

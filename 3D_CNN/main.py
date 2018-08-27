import tensorflow as tf
import os
import argparse
import model
parser = argparse.ArgumentParser(description='3D neural network for HSI classification')
parser.add_argument('--result',dest='result',default='result')
parser.add_argument('--data_path',dest='data_path',default="/data3/zhangjinyang/HSI_Classification/Data")
parser.add_argument('--TFrecords',dest='TFrecords',default='TFrecords')
parser.add_argument('--checkpoint',dest='checkpoint',default='checkpoint')
parser.add_argument('--log',dest='log',default='log')

parser.add_argument('--learning_rate',dest='learning_rate',default=1e-2)
parser.add_argument('--batch_size',dest='batch_size',default=200)
parser.add_argument('--test_batch',dest='test_batch',default=10000)

parser.add_argument('--cube_size',dest='cube_size',default=3)
parser.add_argument('--train_num',dest='train_num',default=200)
parser.add_argument('--dataset',dest='dataset',default='Indian_pines')
parser.add_argument('--iterate_num',dest='iterate_num',default=100000)
parser.add_argument('--reuse',dest='reuse',default=False)

args = parser.parse_args()

def main(_):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 分配50%
    for i in range(1):
        for args.dataset in ['Indian_pines']:# 'PaviaU','Indian_pines','Salinas'
            tf.reset_default_graph()
            with tf.Session(config=config) as sess:
                args.checkpoint = 'checkpoint'+ args.dataset+str(i)
                nc = model.NC(sess,args,id=i)
                nc.data_prepare()
                nc.train()
                nc.test()
                sess.close()

if __name__ == '__main__':
    tf.app.run()

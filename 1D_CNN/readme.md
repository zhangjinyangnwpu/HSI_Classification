## HIS data Classification by 1DCNN classfier

Here, we illustriate PaviaU datasets and use 1DCNN for classification task.
For per category, we selected 200 samples randomly for training, and others for testing.

And the performance I may didn't find the best parameters, and here set the cube size as 1*1. 

### result



PaviaU:

![](pic/decode_map_PaviaU.png)

```
DataSet PaviaU shape is (610, 340, 103)
traindata-ID 1: 200; testdata-ID 1: 6431
traindata-ID 2: 200; testdata-ID 2: 18449
traindata-ID 3: 200; testdata-ID 3: 1899
traindata-ID 4: 200; testdata-ID 4: 2864
traindata-ID 5: 200; testdata-ID 5: 1145
traindata-ID 6: 200; testdata-ID 6: 4829
traindata-ID 7: 200; testdata-ID 7: 1130
traindata-ID 8: 200; testdata-ID 8: 3482
traindata-ID 9: 200; testdata-ID 9: 747
total train 1800, total test 40976
Tensor("Placeholder_1:0", shape=(?, 1, 1, 103), dtype=float32)
Tensor("classifer/conv00/Relu:0", shape=(?, 1, 1, 32, 64), dtype=float32)
Tensor("classifer/conv01/Relu:0", shape=(?, 1, 1, 30, 64), dtype=float32)
Tensor("classifer/conv02/Relu:0", shape=(?, 1, 1, 30, 64), dtype=float32)
Tensor("classifer/conv10/Relu:0", shape=(?, 1, 1, 14, 128), dtype=float32)
Tensor("classifer/conv11/Relu:0", shape=(?, 1, 1, 6, 128), dtype=float32)
Tensor("classifer/conv12/Relu:0", shape=(?, 1, 1, 6, 128), dtype=float32)
Tensor("classifer/conv20/Relu:0", shape=(?, 1, 1, 4, 256), dtype=float32)
Tensor("classifer/conv21/Relu:0", shape=(?, 1, 1, 1, 256), dtype=float32)
Tensor("classifer/conv22/Relu:0", shape=(?, 1, 1, 1, 256), dtype=float32)
Tensor("classifer/global_info/flatten/Reshape:0", shape=(?, 512), dtype=float32)
1 class: ( 5577 / 6431 ) 0.8672057222826932
2 class: ( 17313 / 18449 ) 0.9384248468751694
3 class: ( 1602 / 1899 ) 0.8436018957345972
4 class: ( 2777 / 2864 ) 0.9696229050279329
5 class: ( 1141 / 1145 ) 0.9965065502183406
6 class: ( 4489 / 4829 ) 0.929592048043073
7 class: ( 1061 / 1130 ) 0.9389380530973451
8 class: ( 2936 / 3482 ) 0.8431935669155658
9 class: ( 747 / 747 ) 1.0
confusion matrix:
[[ 5577     1    18     0     4    13    63    87     0]
 [   12 17313     7    76     0   291     0    18     0]
 [  108     0  1602     0     0     0     4   407     0]
 [    0   366     0  2777     0    17     0     0     0]
 [    1     0     0     1  1141     1     0     0     0]
 [   59   759     1     9     0  4489     1    19     0]
 [  531     0     2     0     0     0  1061    15     0]
 [  143    10   269     1     0    18     1  2936     0]
 [    0     0     0     0     0     0     0     0   747]]
total right num: 37643
oa: 0.9186597032409215
aa: 0.9252317320216351
kappa: 0.8916784573500552
Groundtruth map get finished
test end!
decode map get finished
test end!
seg decode map get finished

```

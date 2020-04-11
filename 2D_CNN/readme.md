## HIS data Classification by 2DCNN classfier

Here, we illustriate PaviaU datasets and use 2DCNN for classification task.
For per category, we selected 200 samples randomly for training, and others for testing.

And the performance I may didn't find the best parameters, and here set the cube size as 3*3. 

### result

PaviaU

![](pic/decode_map_PaviaU.png)


```
result
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
Tensor("Placeholder_1:0", shape=(?, 3, 3, 103), dtype=float32)
Tensor("classifer/conv00/Relu:0", shape=(?, 3, 3, 64), dtype=float32)
Tensor("classifer/conv01/Relu:0", shape=(?, 3, 3, 64), dtype=float32)
Tensor("classifer/conv02/Relu:0", shape=(?, 3, 3, 64), dtype=float32)
Tensor("classifer/conv10/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
Tensor("classifer/conv11/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
Tensor("classifer/conv12/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
Tensor("classifer/conv20/Relu:0", shape=(?, 3, 3, 256), dtype=float32)
Tensor("classifer/conv21/Relu:0", shape=(?, 3, 3, 256), dtype=float32)
Tensor("classifer/conv22/Relu:0", shape=(?, 1, 1, 256), dtype=float32)
Tensor("classifer/global_info/flatten/Reshape:0", shape=(?, 512), dtype=float32)
1 class: ( 5999 / 6431 ) 0.9328253770797699
2 class: ( 18089 / 18449 ) 0.9804867472491734
3 class: ( 1806 / 1899 ) 0.9510268562401264
4 class: ( 2797 / 2864 ) 0.9766061452513967
5 class: ( 1145 / 1145 ) 1.0
6 class: ( 4712 / 4829 ) 0.9757713812383516
7 class: ( 1100 / 1130 ) 0.9734513274336283
8 class: ( 3277 / 3482 ) 0.9411257897759908
9 class: ( 743 / 747 ) 0.9946452476572959
confusion matrix:
[[ 5999     0     4     1     0     0    27    41     3]
 [    0 18089     0    53     0   115     0     0     0]
 [   72     0  1806     0     0     0     0   158     0]
 [    2   215     0  2797     0     1     0     0     0]
 [    7     0     1     6  1145     0     0     0     0]
 [   28   144     0     7     0  4712     0     5     1]
 [  149     0     1     0     0     0  1100     1     0]
 [  174     1    87     0     0     1     3  3277     0]
 [    0     0     0     0     0     0     0     0   743]]
total right num: 39668
oa: 0.9680788754392815
aa: 0.9695487635473037
kappa: 0.9571692958968724
Groundtruth map get finished
test end!
decode map get finished
test end!
seg decode map get finished

```

# 1D convolution network for HSI Classification
#### You can try more iterate number for a better performence, here I just list a result I run. 
I choise 200 samples form per class, and take the other as test dataset. For more detail, read code please, and any question, contact me with no hesitate.

- 30000 iteratation for **PaviaU** with batch size 200
```angular2html
1 class: ( 5445 / 6431 ) 0.8466801430570673
2 class: ( 16722 / 18449 ) 0.9063905902758957
3 class: ( 1673 / 1899 ) 0.8809899947340706
4 class: ( 2749 / 2864 ) 0.9598463687150838
5 class: ( 1142 / 1145 ) 0.9973799126637555
6 class: ( 4359 / 4829 ) 0.9026713605301304
7 class: ( 1081 / 1130 ) 0.9566371681415929
8 class: ( 2753 / 3482 ) 0.7906375646180356
9 class: ( 746 / 747 ) 0.998661311914324
confusion matrix:
[[ 5445     0    24     0     2     2    43   107     1]
 [   17 16722    10    92     1   411     0    32     0]
 [  126    10  1673     0     0     0     3   553     0]
 [    4   537     0  2749     0    20     0     0     0]
 [    0     2     0     1  1142     4     0     1     0]
 [   50  1151     1    22     0  4359     2    26     0]
 [  643     0     8     0     0     2  1081    10     0]
 [  146    27   183     0     0    31     1  2753     0]
 [    0     0     0     0     0     0     0     0   746]]
total right num: 36670
oa: 0.8949140960562281
aa: 0.9155438238499952
kappa: 0.8608634090854034
```

- 30000 iteratation for **Salinas** with batch size 200
```angular2html
1 class: ( 1735 / 1809 ) 0.9590934217799889
2 class: ( 3516 / 3526 ) 0.9971639251276234
3 class: ( 1758 / 1776 ) 0.9898648648648649
4 class: ( 1187 / 1194 ) 0.9941373534338358
5 class: ( 2427 / 2478 ) 0.9794188861985472
6 class: ( 3756 / 3759 ) 0.9992019154030327
7 class: ( 3353 / 3379 ) 0.9923054158034922
8 class: ( 8664 / 11071 ) 0.782585132327703
9 class: ( 5984 / 6003 ) 0.9968349158753956
10 class: ( 2987 / 3078 ) 0.9704353476283301
11 class: ( 852 / 868 ) 0.9815668202764977
12 class: ( 1692 / 1727 ) 0.9797336421540244
13 class: ( 713 / 716 ) 0.9958100558659218
14 class: ( 868 / 870 ) 0.9977011494252873
15 class: ( 5093 / 7068 ) 0.7205715902659876
16 class: ( 1593 / 1607 ) 0.9912881144990666
confusion matrix:
[[1735    0    0    0    0    0    0    0    0    0    0    0    0    0
     0    0]
 [  73 3516    0    0    0    0    0    0    0    0    0    0    0    0
     0    8]
 [   0    0 1758    0    3    0    0    0    0   12    0    0    0    0
     0    0]
 [   0    0    0 1187   18    0    0    0    0    1    0    0    0    0
     0    0]
 [   0    0    6    7 2427    0    0    0    0    1    3    0    0    0
     1    0]
 [   0    0    0    0    1 3756    0    0    0    0    0    0    0    0
     0    0]
 [   0    0    0    0    0    0 3353   16    0    0    0    0    0    0
    15    1]
 [   0    0    0    0    0    0    5 8664    3    5    0    0    0    0
  1841    0]
 [   0    0    0    0    1    0    0    0 5984   19    0    1    0    0
     0    0]
 [   0    0   12    0   28    1    2  248   13 2987    9   34    0    2
    81    1]
 [   0    0    0    0    0    0    0    0    0    8  852    0    0    0
     0    0]
 [   0    0    0    0    0    0    0    0    0    0    4 1692    0    0
     0    0]
 [   0    0    0    0    0    0    2    0    0    0    0    0  713    0
     0    0]
 [   0    7    0    0    0    2    8   10    3   21    0    0    3  868
     3    1]
 [   0    0    0    0    0    0    0 2130    0    9    0    0    0    0
  5093    3]
 [   1    3    0    0    0    0    9    3    0   15    0    0    0    0
    34 1593]]
total right num: 46178
oa: 0.9067132674900351
aa: 0.9579820344331
kappa: 0.8957019655512752
```

- 50000 iteratation for **IndianPines** with batch size 200,
here we choise the most large 9 classes in this dataset.
```angular2html
feature map shape
Tensor("ExpandDims:0", shape=(?, 220, 1), dtype=float32)
Tensor("classifer/conv0/Relu:0", shape=(?, 71, 16), dtype=float32)
Tensor("classifer/conv1/Relu:0", shape=(?, 35, 32), dtype=float32)
Tensor("classifer/conv2/Relu:0", shape=(?, 17, 64), dtype=float32)
Tensor("classifer/global_info/flatten/Reshape:0", shape=(?, 9), dtype=float32)

1 class: ( 1123 / 1228 ) 0.9144951140065146
2 class: ( 493 / 630 ) 0.7825396825396825
3 class: ( 272 / 283 ) 0.9611307420494699
4 class: ( 526 / 530 ) 0.9924528301886792
5 class: ( 278 / 278 ) 1.0
6 class: ( 589 / 772 ) 0.7629533678756477
7 class: ( 1603 / 2255 ) 0.7108647450110864
8 class: ( 361 / 393 ) 0.9185750636132316
9 class: ( 1054 / 1065 ) 0.9896713615023475
confusion matrix:
[[1123   70    1    0    0   93  304   11    1]
 [  16  493    0    0    0    5  100    9    0]
 [   0    0  272    1    0    1   12    0    7]
 [   4    1    0  526    0    1    0    1    3]
 [   0    0    1    0  278    0    1    0    0]
 [  36    5    4    1    0  589  193    3    0]
 [  47   41    1    1    0   82 1603    7    0]
 [   1   20    2    0    0    1   42  361    0]
 [   1    0    2    1    0    0    0    1 1054]]
total right num: 6299
oa: 0.8473231100349744
aa: 0.8925203229762955
kappa: 0.8185579819605011

```
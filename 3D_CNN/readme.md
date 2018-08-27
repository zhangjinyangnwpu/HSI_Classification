# 3D convolution network for HSI Classification
#### You can try more iterate number for a better performence, here I just list a result I run. 
I choise 200 samples form per class, and take the other as test dataset. For more detail, read code please, and any question, contact me with no hesitate.

- 100000 iteratation for **PaviaU** with batch size 200
```angular2html
Tensor("Placeholder:0", shape=(?, 3, 3, 103, 1), dtype=float32)
Tensor("inference_10/network/conv0/Relu:0", shape=(?, 3, 3, 52, 16), dtype=float32)
Tensor("inference_10/network/conv1/Relu:0", shape=(?, 1, 3, 25, 32), dtype=float32)
Tensor("inference_10/network/conv2/Relu:0", shape=(?, 1, 1, 12, 64), dtype=float32)
Tensor("inference_10/network/conv3/Relu:0", shape=(?, 1, 1, 6, 128), dtype=float32)
Tensor("inference_10/network/score/conv3d/BiasAdd:0", shape=(?, 1, 1, 1, 9), dtype=float32)
Tensor("inference_10/network/flatten/Reshape:0", shape=(?, 9), dtype=float32)
Loading model ...
load successful...
end!
[[ 6094    28    61     1     0     7    96   144     0]
 [   10 17321     2   325     0   784     0     7     0]
 [    7    20  1766     0     0     0     0   106     0]
 [   19    64     0  2756     0    20     0     5     0]
 [    0     0     0     0  1144     1     0     0     0]
 [    3   162     0    28     1  4620     0    15     0]
 [   20     0     2     0     0     0  1102     6     0]
 [   38    30   151     0     0    19     0  3244     0]
 [    3     0     0     0     0     0     1     0   743]]
( 6094 / 6431 ) 0.9475975742497279
( 17321 / 18449 ) 0.9388584747140767
( 1766 / 1899 ) 0.9299631384939442
( 2756 / 2864 ) 0.9622905027932961
( 1144 / 1145 ) 0.9991266375545852
( 4620 / 4829 ) 0.9567198177676538
( 1102 / 1130 ) 0.9752212389380531
( 3244 / 3482 ) 0.9316484778862723
( 743 / 747 ) 0.9946452476572959
38790 / 40976
accuracy: 0.9466516985552519
kappa: 0.9288839104210191

```

- 100000 iteratation for **Salinas** with batch size 200
```angular2html
Tensor("Placeholder:0", shape=(?, 3, 3, 204, 1), dtype=float32)
Tensor("inference_10/network/conv0/Relu:0", shape=(?, 3, 3, 102, 16), dtype=float32)
Tensor("inference_10/network/conv1/Relu:0", shape=(?, 1, 3, 50, 32), dtype=float32)
Tensor("inference_10/network/conv2/Relu:0", shape=(?, 1, 1, 24, 64), dtype=float32)
Tensor("inference_10/network/conv3/Relu:0", shape=(?, 1, 1, 12, 128), dtype=float32)
Tensor("inference_10/network/score/conv3d/BiasAdd:0", shape=(?, 1, 1, 1, 16), dtype=float32)
Tensor("inference_10/network/flatten/Reshape:0", shape=(?, 16), dtype=float32)
Loading model ...
load successful...
end!
[[1778    2    0    0    0    0    0    0    0    0    0    0    0    0
     0   29]
 [   0 3522    0    0    0    0    0    0    0    0    0    0    0    1
     0    3]
 [   0    0 1756    0    4    0    0    0   11    5    0    0    0    0
     0    0]
 [   0    0    1 1190    2    0    0    0    0    0    0    0    1    0
     0    0]
 [   0    0   20    5 2439    0    0    0    0    6    2    0    0    0
     0    6]
 [   0    0    0    0    0 3758    0    0    0    0    0    0    0    1
     0    0]
 [   0    1    0    0    0    0 3361    0    0    0    0    0    0   16
     0    1]
 [   0    0    0    2    0   16    3 9081    1  190    0    0    5    2
  1703   68]
 [   0    0    0    0    0    0    0    0 5982    9    2    4    0    0
     0    6]
 [   0    0   54    1    0    1    0   29   31 2896   12    9    1    2
     4   38]
 [   0    0    0    0    0    0    0    2    0    0  845    3    0    0
     0   18]
 [   0    0    0    0    0    0    0    0    0    0    0 1711    0    0
     0   16]
 [   0    0    0    0    0    0    0    0    0    0    0    0  710    1
     0    5]
 [   0    0    0    0    0    0    0    0    0    2    0    0    2  859
     0    7]
 [   0    0    0    1    0    0    0 1555    0    8    3   12    3    0
  5367  119]
 [   0    0    0    0    0    0    0    2    0    0    0    1    0    0
     0 1604]]
( 1778 / 1809 ) 0.9828634604754007
( 3522 / 3526 ) 0.9988655700510494
( 1756 / 1776 ) 0.9887387387387387
( 1190 / 1194 ) 0.9966499162479062
( 2439 / 2478 ) 0.9842615012106537
( 3758 / 3759 ) 0.9997339718010109
( 3361 / 3379 ) 0.9946729801716484
( 9081 / 11071 ) 0.820251106494445
( 5982 / 6003 ) 0.9965017491254373
( 2896 / 3078 ) 0.9408706952566601
( 845 / 868 ) 0.9735023041474654
( 1711 / 1727 ) 0.9907353792704111
( 710 / 716 ) 0.9916201117318436
( 859 / 870 ) 0.9873563218390805
( 5367 / 7068 ) 0.7593378607809848
( 1604 / 1607 ) 0.9981331673926571
46859 / 50929
accuracy: 0.9200848239706257
kappa: 0.9106477504227228
```

- 100000 iteratation for **IndianPines** with batch size 200
```angular2html
Tensor("Placeholder:0", shape=(?, 3, 3, 220, 1), dtype=float32)
Tensor("inference_10/network/conv0/Relu:0", shape=(?, 3, 3, 110, 16), dtype=float32)
Tensor("inference_10/network/conv1/Relu:0", shape=(?, 1, 3, 54, 32), dtype=float32)
Tensor("inference_10/network/conv2/Relu:0", shape=(?, 1, 1, 26, 64), dtype=float32)
Tensor("inference_10/network/conv3/Relu:0", shape=(?, 1, 1, 13, 128), dtype=float32)
Tensor("inference_10/network/score/conv3d/BiasAdd:0", shape=(?, 1, 1, 1, 9), dtype=float32)
Tensor("inference_10/network/flatten/Reshape:0", shape=(?, 9), dtype=float32)
Loading model ...
load successful...
end!
[[1048   62    0    1    0   51   54   11    1]
 [  50  531    0    1    0    5   20   23    0]
 [   1    0  277    1    0    0    0    4    0]
 [   0    0    2  527    0    0    0    0    1]
 [   0    0    0    0  278    0    0    0    0]
 [  22    5    2    2    0  701   37    3    0]
 [  90   88    8    6    0  176 1810   77    0]
 [   3    3    1    3    0    2    3  378    0]
 [   0    2    1    4    0    0    0    0 1058]]
( 1048 / 1228 ) 0.8534201954397395
( 531 / 630 ) 0.8428571428571429
( 277 / 283 ) 0.9787985865724381
( 527 / 530 ) 0.9943396226415094
( 278 / 278 ) 1.0
( 701 / 772 ) 0.9080310880829016
( 1810 / 2255 ) 0.802660753880266
( 378 / 393 ) 0.9618320610687023
( 1058 / 1065 ) 0.9934272300469483
6608 / 7434
accuracy: 0.8888888888888888
kappa: 0.8679621383966392
```
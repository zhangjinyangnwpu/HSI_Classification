# 2D convolution network for HSI Classification
#### You can try more iterate number for a better performence, here I just list a result I run. 
I choise 200 samples form per class, and take the other as test dataset. For more detail, read code please, and any question, contact me with no hesitate.
Here, the cube size is 3, you can set more large cube size for better performence, and just change one hyperparameter cube_size in args
All iterate number is 100000, and batch_size is 200

PaviaU
```markdown
ensor("Placeholder:0", shape=(?, 3, 3, 103), dtype=float32)
Tensor("classifer/conv0/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
Tensor("classifer/conv1/Relu:0", shape=(?, 1, 3, 256), dtype=float32)
Tensor("classifer/conv2/Relu:0", shape=(?, 1, 1, 512), dtype=float32)
Tensor("classifer/global_info/flatten/Reshape:0", shape=(?, 9), dtype=float32)
test end!
1 class: ( 5980 / 6431 ) 0.9298709376457782
2 class: ( 17186 / 18449 ) 0.9315410049325167
3 class: ( 1773 / 1899 ) 0.933649289099526
4 class: ( 2723 / 2864 ) 0.950768156424581
5 class: ( 1145 / 1145 ) 1.0
6 class: ( 4398 / 4829 ) 0.9107475667840133
7 class: ( 1063 / 1130 ) 0.9407079646017699
8 class: ( 3220 / 3482 ) 0.9247558874210224
9 class: ( 744 / 747 ) 0.9959839357429718
confusion matrix:
[[ 5980     0    12     3     0     4    63   121     3]
 [    1 17186     0    78     0   350     0     3     0]
 [   53    90  1773     0     0     1     2   125     0]
 [    4   297     0  2723     0     6     0     0     0]
 [    8     0     0     0  1145     0     0     0     0]
 [   10   858     7    50     0  4398     0    11     0]
 [  219     0     0     0     0     0  1063     2     0]
 [  142     1   104     0     0    58     2  3220     0]
 [   14    17     3    10     0    12     0     0   744]]
total right num: 38232
oa: 0.933033971105037
aa: 0.94644719362802
kappa: 0.9107809983181837
Groundtruth map get finished
test end!
decode map get finished

```
Salinas
```markdown
Tensor("Placeholder:0", shape=(?, 3, 3, 204), dtype=float32)
Tensor("classifer/conv0/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
Tensor("classifer/conv1/Relu:0", shape=(?, 1, 3, 256), dtype=float32)
Tensor("classifer/conv2/Relu:0", shape=(?, 1, 1, 512), dtype=float32)
Tensor("classifer/global_info/flatten/Reshape:0", shape=(?, 16), dtype=float32)
test end!
1 class: ( 1783 / 1809 ) 0.9856274184632393
2 class: ( 3516 / 3526 ) 0.9971639251276234
3 class: ( 1772 / 1776 ) 0.9977477477477478
4 class: ( 1192 / 1194 ) 0.998324958123953
5 class: ( 2459 / 2478 ) 0.9923325262308313
6 class: ( 3757 / 3759 ) 0.9994679436020218
7 class: ( 3367 / 3379 ) 0.9964486534477656
8 class: ( 9100 / 11071 ) 0.8219673019600758
9 class: ( 5986 / 6003 ) 0.997168082625354
10 class: ( 2890 / 3078 ) 0.9389213775178687
11 class: ( 860 / 868 ) 0.9907834101382489
12 class: ( 1711 / 1727 ) 0.9907353792704111
13 class: ( 708 / 716 ) 0.9888268156424581
14 class: ( 854 / 870 ) 0.9816091954022989
15 class: ( 5700 / 7068 ) 0.8064516129032258
16 class: ( 1586 / 1607 ) 0.9869321717485999
confusion matrix:
[[1783    0    0    0    0    0    0    0    0    1    0    1    0    0
     1    1]
 [  12 3516    0    0    0    1    0    0    0    0    0    0    0    0
     0    0]
 [   0    0 1772    0    4    0    0    0    0   40    0    0    0    0
     2    0]
 [   0    0    0 1192    7    0    0    0    0    0    1    0    0    0
     0    0]
 [   0    0    4    2 2459    0    0    0    0    0    0    0    0    0
     0    0]
 [   0    0    0    0    0 3757    0    0    0    0    0    0    0    0
     0    0]
 [   0    6    0    0    0    0 3367   11    0    1    0    0    0    0
     0    5]
 [   0    0    0    0    0    0    1 9100    0   22    0    0    0    3
  1265    0]
 [   0    0    0    0    0    0    0    2 5986   17    1    1    1    0
     1    0]
 [   0    0    0    0    0    0    0  259    2 2890    1    0    0    1
    28    0]
 [   0    0    0    0    5    0    0    8    9   57  860    6    2    1
     4    0]
 [   1    0    0    0    3    0    0    9    1    6    1 1711    0    2
    15    0]
 [   9    0    0    0    0    0    0    7    0    2    0    0  708    8
    11    2]
 [   2    0    0    0    0    0   11    7    2   17    0    5    5  854
    11    1]
 [   0    0    0    0    0    0    0 1668    3    5    4    3    0    0
  5700   12]
 [   2    4    0    0    0    1    0    0    0   20    0    0    0    1
    30 1586]]
total right num: 47241
oa: 0.9275854621139233
aa: 0.9669067824969827
kappa: 0.9190666315059712

```
Indian_pines
```markdown
Tensor("Placeholder:0", shape=(?, 3, 3, 220), dtype=float32)
Tensor("classifer/conv0/Relu:0", shape=(?, 3, 3, 128), dtype=float32)
Tensor("classifer/conv1/Relu:0", shape=(?, 1, 3, 256), dtype=float32)
Tensor("classifer/conv2/Relu:0", shape=(?, 1, 1, 512), dtype=float32)
Tensor("classifer/global_info/flatten/Reshape:0", shape=(?, 9), dtype=float32)
1 class: ( 998 / 1228 ) 0.8127035830618893
2 class: ( 563 / 630 ) 0.8936507936507937
3 class: ( 273 / 283 ) 0.9646643109540636
4 class: ( 523 / 530 ) 0.9867924528301887
5 class: ( 278 / 278 ) 1.0
6 class: ( 718 / 772 ) 0.9300518134715026
7 class: ( 1786 / 2255 ) 0.7920177383592018
8 class: ( 355 / 393 ) 0.9033078880407125
9 class: ( 1047 / 1065 ) 0.9830985915492958
confusion matrix:
[[ 998   12    3    0    0   18  129   12    0]
 [  50  563    1    0    0    5   96    2    0]
 [   0    1  273    0    0    2   18    2    5]
 [   5    0    3  523    0    4    1    2   13]
 [   0    0    0    0  278    0    0    0    0]
 [  61    3    0    2    0  718  180    8    0]
 [  58   17    0    0    0   14 1786   11    0]
 [  56   34    1    0    0   11   44  355    0]
 [   0    0    2    5    0    0    1    1 1047]]
total right num: 6541
oa: 0.8798762442830239
aa: 0.918476352435294
kappa: 0.8575235063093629
Groundtruth map get finished
test end!
decode map get finished

```
## HIS data Classification by SVM(支撑向量机) classfier

Here, I illustriate University of Pavia dataset and use svm for classification task.
For per category, we selected **200** samples randomly for training, and others for testing.

I may didn't find the best parameters, and here set the kernel function as rbf, gamma is 1, and the penalize as 100 .

Once run result.
```
### result
1 class: ( 5527.0 / 6431.0 ) 0.8594308816669258
2 class: ( 17143.0 / 18449.0 ) 0.9292102552983902
3 class: ( 1656.0 / 1899.0 ) 0.8720379146919431
4 class: ( 2744.0 / 2864.0 ) 0.9581005586592178
5 class: ( 1144.0 / 1145.0 ) 0.9991266375545852
6 class: ( 4409.0 / 4829.0 ) 0.9130254711120315
7 class: ( 1065.0 / 1130.0 ) 0.9424778761061947
8 class: ( 2957.0 / 3482.0 ) 0.8492245835726594
9 class: ( 745.0 / 747.0 ) 0.9973226238286479
confusion matrix:
[[ 5527     0    22     0     1     9    63    52     2]
 [   14 17143    11    92     0   329     0     7     0]
 [   98     0  1656     0     0     3     1   443     0]
 [    0   243     0  2744     0     4     0     0     0]
 [    4     0     0     9  1144     5     0     0     0]
 [   35  1041     6    19     0  4409     0    20     0]
 [  580     0     4     0     0     0  1065     3     0]
 [  173    22   200     0     0    70     1  2957     0]
 [    0     0     0     0     0     0     0     0   745]]
total right num: 37390.0
total test num: 40976.0
Overall accuracy: 0.9124853572823116
Average accuracy: 0.9244396447211773
Kappa: 0.8836376165846425
decode map get finished
```
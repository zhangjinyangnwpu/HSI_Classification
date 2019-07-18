## HIS data Classification by SVM(支撑向量机) classfier

Here, I illustriate University of Pavia dataset and use knn for classification task.
For per category, we selected **100** samples randomly for training, and others for testing.

I may didn't find the best parameters, and here set the kernel function as rbf, gamma is 1, and the penalize as 100 .

Once run result.
```
### result
1 class: ( 5482.0 / 6531.0 ) 0.8393814117286786
2 class: ( 16346.0 / 18549.0 ) 0.8812334896759934
3 class: ( 1543.0 / 1999.0 ) 0.7718859429714857
4 class: ( 2811.0 / 2964.0 ) 0.9483805668016194
5 class: ( 1235.0 / 1245.0 ) 0.9919678714859438
6 class: ( 4532.0 / 4929.0 ) 0.9194562791641306
7 class: ( 1161.0 / 1230.0 ) 0.9439024390243902
8 class: ( 2910.0 / 3582.0 ) 0.8123953098827471
9 class: ( 846.0 / 847.0 ) 0.9988193624557261
confusion matrix:
[[ 5482     2   205     0     4     7    64    98     1]
 [   32 16346     6   136     1   317     0    15     0]
 [   66     6  1543     0     0     1     0   540     0]
 [    2   288     0  2811     0     3     0     0     0]
 [   25     0     0     1  1235     4     0     0     0]
 [   52  1894     3    16     1  4532     0    16     0]
 [  664     0     4     0     0     0  1161     3     0]
 [  208    13   238     0     4    65     5  2910     0]
 [    0     0     0     0     0     0     0     0   846]]
total right num: 36866.0
total test num: 41876.0
Overall accuracy: 0.8803610660043939
Average accuracy: 0.9008247414656351
Kappa: 0.8434578905246909
decode map get finished.
```
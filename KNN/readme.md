## HIS data Classification by knn(k近邻) classfier

Here, I illustriate University of Pavia dataset and use knn for classification task.
For per category, we selected **100** samples randomly for training, and others for testing.

I may didn't find the best parameters, and here set the number of neighbor as 5.

Once run result.
```
### result
1 class: ( 4249.0 / 6531.0 ) 0.6505894962486602
2 class: ( 12776.0 / 18549.0 ) 0.6887702841123511
3 class: ( 1480.0 / 1999.0 ) 0.7403701850925463
4 class: ( 2786.0 / 2964.0 ) 0.9399460188933874
5 class: ( 1233.0 / 1245.0 ) 0.9903614457831326
6 class: ( 3612.0 / 4929.0 ) 0.7328058429701765
7 class: ( 1138.0 / 1230.0 ) 0.9252032520325203
8 class: ( 2602.0 / 3582.0 ) 0.7264098269123395
9 class: ( 846.0 / 847.0 ) 0.9988193624557261
confusion matrix:
[[ 4249     1    37     0     1     7    37   114     1]
 [   25 12776    11   158     1  1228     0    20     0]
 [  585     2  1480     0     3    12    47   815     0]
 [    0  1380     0  2786     0     1     0     0     0]
 [   24     0     0     3  1233     6     0     0     0]
 [   69  4375     3    17     2  3612     1    17     0]
 [ 1331     0    16     0     1     0  1138    14     0]
 [  248    15   452     0     4    63     7  2602     0]
 [    0     0     0     0     0     0     0     0   846]]
total right num: 30722.0
total test num: 41876.0
Overall accuracy: 0.7336421816792434
Average accuracy: 0.8214750793889821
Kappa: 0.6627916544158474
decode map get finished.
```
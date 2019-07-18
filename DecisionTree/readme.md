## HIS data Classification by DecisionTree(决策树) classfier

Here, I illustriate University of Pavia dataset and use decision tree for classification task.
For per category, we selected **100** samples randomly for training, and others for testing.

I may didn't find the best parameters, and here set the criterion as gini and max_features as auto.
It seem that the decision tree not suitable for image feature classification. 

Once run result.
```
### result
1 class: ( 4705.0 / 6531.0 ) 0.720410350635431
2 class: ( 11247.0 / 18549.0 ) 0.6063399644185671
3 class: ( 1172.0 / 1999.0 ) 0.5862931465732867
4 class: ( 2657.0 / 2964.0 ) 0.8964237516869096
5 class: ( 1236.0 / 1245.0 ) 0.9927710843373494
6 class: ( 3568.0 / 4929.0 ) 0.7238790829782917
7 class: ( 1040.0 / 1230.0 ) 0.8455284552845529
8 class: ( 2410.0 / 3582.0 ) 0.6728084868788387
9 class: ( 847.0 / 847.0 ) 1.0
confusion matrix:
[[ 4705     7   188     0     0   100   151   130     0]
 [   10 11247     8   229     0  1040     0    20     0]
 [  540     0  1172     0     0     1    19   947     0]
 [    2  1658     0  2657     0   169     0     0     0]
 [   34     0     9     2  1236    17     0     6     0]
 [   72  5623     9    75     4  3568     1    51     0]
 [  938     1    31     0     1     0  1040    18     0]
 [  229    13   582     1     4    34    19  2410     0]
 [    1     0     0     0     0     0     0     0   847]]
total right num: 28882.0
total test num: 41876.0
Overall accuracy: 0.6897029324672843
Average accuracy: 0.7827171469770252
Kappa: 0.6126083093443351
decode map get finished
```
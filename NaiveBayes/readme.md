## HIS data Classification by NaiveBayes(朴素贝叶斯) classfier

Here, I illustriate University of Pavia dataset and use naive bayes for classification task.
For per category, we selected **100** samples randomly for training, and others for testing.

I may didn't find the best parameters, and here set the penatly set as 'l2', and the solver set as 'newton-cg'(拟牛顿法的一种).
noted solver as liblinear will be very slow, because sklearn just support one-vs-multi for LR with linear solver

Once run result.
```
### result
1 class: ( 4148.0 / 6531.0 ) 0.6351247894656255
2 class: ( 16259.0 / 18549.0 ) 0.8765432098765432
3 class: ( 1765.0 / 1999.0 ) 0.8829414707353677
4 class: ( 2743.0 / 2964.0 ) 0.9254385964912281
5 class: ( 1240.0 / 1245.0 ) 0.9959839357429718
6 class: ( 2868.0 / 4929.0 ) 0.5818624467437614
7 class: ( 1155.0 / 1230.0 ) 0.9390243902439024
8 class: ( 531.0 / 3582.0 ) 0.14824120603015076
9 class: ( 846.0 / 847.0 ) 0.9988193624557261
confusion matrix:
[[ 4148     0    78     0     0    40    68   754     0]
 [   24 16259     2   201     2  1899     0     5     0]
 [  541    81  1765     0     0    11     3  1605     0]
 [    3   758     0  2743     0    11     0     0     0]
 [    8     0     0     1  1240    13     0     0     0]
 [   65  1421     6    10     1  2868     4   544     0]
 [ 1474     0    36     0     0     6  1155   143     1]
 [  215    30   111     7     2    81     0   531     0]
 [   53     0     1     2     0     0     0     0   846]]
total right num: 31555.0
total test num: 41876.0
Overall accuracy: 0.7535342439583532
Average accuracy: 0.7759977119761419
Kappa: 0.6743618932773663
decode map get finished
```
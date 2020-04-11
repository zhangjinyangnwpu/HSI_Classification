## HIS data Classification by knn(k近邻) classfier

Here, I illustriate University of Pavia dataset and use knn for classification task.
For per category, we selected **200** samples randomly for training, and others for testing.

I may didn't find the best parameters, and here set the number of neighbor as 5.

Once run result.
```
### result
1 class: ( 4679.0 / 6431.0 ) 0.7275695848235111
2 class: ( 14208.0 / 18449.0 ) 0.7701230418992899
3 class: ( 1536.0 / 1899.0 ) 0.8088467614533965
4 class: ( 2731.0 / 2864.0 ) 0.9535614525139665
5 class: ( 1139.0 / 1145.0 ) 0.994759825327511
6 class: ( 3477.0 / 4829.0 ) 0.7200248498653966
7 class: ( 1047.0 / 1130.0 ) 0.9265486725663716
8 class: ( 2600.0 / 3482.0 ) 0.7466973004020678
9 class: ( 747.0 / 747.0 ) 1.0
confusion matrix:
[[ 4679     0    52     0     3     6    53    68     0]
 [    7 14208     3   118     0  1288     0    12     0]
 [  461    10  1536     0     0     2    27   761     0]
 [    0  1306     0  2731     0     5     0     0     0]
 [   27     0     0     4  1139     7     0     0     0]
 [   87  2916     5    10     1  3477     1    23     0]
 [  944     0     7     0     0     0  1047    18     0]
 [  226     9   296     0     2    44     2  2600     0]
 [    0     0     0     1     0     0     0     0   747]]
total right num: 32164.0
total test num: 40976.0
Overall accuracy: 0.7849472862163217
Average accuracy: 0.849792387650168
Kappa: 0.7213188818212138
decode map get finished
```
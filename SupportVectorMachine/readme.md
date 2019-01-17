## HIS data Classification by knn(k近邻) classfier

Here, we illustriate some datasets and use svm for classification task.
For per category, we selected 200 samples(15 for the class number less than 200 in Indian_pines) randomly for training, and others for testing.

And the performance I may didn't find the best parameters, and her set the number of neighbor as 5.

### result
- Indian_pines
    - decode map
    ![decode_map](./result/decode_mapIndian_pines.png)


- PaviaU
    - decode map
    ![decode_map](./result/decode_mapPaviaU.png)

- Salinas
    - decode map
    ![decode_map](./result/decode_mapSalinas.png)

training number 200 per class, neighbour k 5

|       | Indian Pines | PaviaU   | Salinas |
|:-----:|:------------:|:--------:|:-------:|
|OA     |    82.81     |  90.57   |    91.19|
|AA     |    86.81     |   92.30  |95.83    |
|Kappa  |    80.00     |  87.51   |90.14    |
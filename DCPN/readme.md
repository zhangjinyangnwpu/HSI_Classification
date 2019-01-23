## Deep Cube Pair Network for Hyperspectral Image Classification
Paper link  [deep cube pair network for hyperspectral image classification](www.baidu.com)
#### Abstruct
Advanced classification methods, which can fully utilize the 3D characteristic of hyperspectral image (HSI) and generalize well to the test data given only limited labeled training samples (i.e., small training dataset), have long been the research objective for HSI classification problem. Witnessing the success of deep-learning-based methods, a cube-pair-based convolutional neural networks (CNN) classification architecture is proposed to cope this objective in this study, where cube-pair is used to address the small training dataset problem as well as preserve the 3D local structure of HSI data. Within this architecture, a 3D fully convolutional network is further modeled, which has less parameters compared with traditional CNN. Provided the same amount of training samples, the modeled network can go deeper than traditional CNN and thus has superior generalization ability. Experimental results on several HSI datasets demonstrate that the proposed method has superior classification results compared with other state-of-the-art competing methods.

#### Framework
![Framework](./pic/train_test_framework.png)

#### result
    The experiment is finished by tensorflow in python3.
    

training number 200 per class

|       | Indian Pines | PaviaU   | Salinas |
|:-----:|:------------:|:--------:|:-------:|
|OA     |         |     |    |
|AA     |         |     |    |
|Kappa  |         |     |    |
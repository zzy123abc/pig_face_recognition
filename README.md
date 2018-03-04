# pig_face_recognition

京东JDD猪脸识别比赛

pytorch-baseline 

1.运行环境

TeslaK20c集群单节点双卡 Red Hat 4.4.7-3 Python 2.7.13 cuda 8.0 cudnn 5.0 pytorch 0.3.0

2.从视频中截取出猪

用yolo-9000算法，人工打label后，对ffmpeg提取出的视频帧进行猪的目标检测，框出猪的主体部分，为后续分类做基础。

3.数据预处理

把下载的数据集预处理，生成torchvision.datasets.ImageFolder接口需要的文件夹格式，并使用torchvision.transforms中的方法进行数据增强，具体采用了RandomResizedCrop、RandomHorizontalFlip、ColorJitter以及Normalize.

4.train from scratch or fitune from imagenet

尝试多种resnet和densenet网络，最后选择较好的结果进行平均融合。

5.测试结果

对测试数据集进行前向运算得到预测分类概率并保存。

6.提交结果

转换分数格式为所需格式并提交结果。

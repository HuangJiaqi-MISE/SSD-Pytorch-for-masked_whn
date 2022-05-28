# SSD：Single-Shot MultiBox Detector目标检测模型在Masked-Face-Dataset数据集中的训练效果


## Ⅰ. 所需环境

```bash
scipy==1.2.1
numpy==1.17.0
matplotlib==3.1.2
opencv_python==4.1.2.30
torch==1.2.0
torchvision==0.4.0
tqdm==4.60.0
Pillow==8.2.0
h5py==2.10.0
```

## Ⅱ. 训练步骤

###数据集的准备

本文使用VOC格式进行训练，训练前需要自己制作好数据集，我们使用labelimg对数据集中的所有图片样本进行标注。

训练前将标签文件存放在VOCdevkit文件夹下的VOC2007文件夹下的Annotation中。  
训练前将图片文件存放在VOCdevkit文件夹下的VOC2007文件夹下的JPEGImages中。   

###数据集的处理

在完成数据集的摆放之后，我们需要利用voc_annotation.py获得训练用的2007_train.txt和2007_val.txt。   
修改voc_annotation.py里面的参数。第一次训练可以仅修改classes_path，classes_path用于指向检测类别所对应的txt。   
训练自己的数据集时，可以自己建立一个cls_classes.txt，里面写自己所需要区分的类别。

model_data/cls_classes.txt文件内容为：      
```python
mask
ummask
```

修改voc_annotation.py中的classes_path，使其对应cls_classes.txt，并运行voc_annotation.py。  

###开始网络训练

训练的参数配置信息在train.py中，其中最重要的部分依然是train.py里的classes_path。  
classes_path用于指向检测类别所对应的txt，这个txt和voc_annotation.py里面的txt一样，训练自己的数据集必须要修改。  
修改完classes_path后就可以运行train.py开始训练了，在训练多个epoch后，权值会生成在logs文件夹中。  

###训练结果预测

训练结果预测需要用到两个文件，分别是ssd.py和predict.py。在ssd.py里面修改model_path以及classes_path。  
**model_path指向训练好的权值文件，在logs文件夹里。  
classes_path指向检测类别所对应的txt。**  
完成修改后就可以运行predict.py进行检测了。运行后输入图片路径即可检测。  

## 预测步骤
### 使用自己训练的权重
1. 按照训练步骤训练。  
2. 在ssd.py文件里面，在如下部分修改model_path和classes_path使其对应训练好的文件；**model_path对应logs文件夹下面的权值文件，classes_path是model_path对应分的类**。  
```python
_defaults = {
    #--------------------------------------------------------------------------#
    #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
    #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
    #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
    #--------------------------------------------------------------------------#
    "model_path"        : 'model_data/ssd_weights.pth',
    "classes_path"      : 'model_data/voc_classes.txt',
    #---------------------------------------------------------------------#
    #   用于预测的图像大小，和train时使用同一个即可
    #---------------------------------------------------------------------#
    "input_shape"       : [300, 300],
    #-------------------------------#
    #   主干网络的选择
    #   vgg或者mobilenetv2
    #-------------------------------#
    "backbone"          : "vgg",
    #---------------------------------------------------------------------#
    #   只有得分大于置信度的预测框会被保留下来
    #---------------------------------------------------------------------#
    "confidence"        : 0.5,
    #---------------------------------------------------------------------#
    #   非极大抑制所用到的nms_iou大小
    #---------------------------------------------------------------------#
    "nms_iou"           : 0.45,
    #---------------------------------------------------------------------#
    #   用于指定先验框的大小
    #---------------------------------------------------------------------#
    'anchors_size'      : [30, 60, 111, 162, 213, 264, 315],
    #---------------------------------------------------------------------#
    #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
    #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
    #---------------------------------------------------------------------#
    "letterbox_image"   : False,
    #-------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    #-------------------------------#
    "cuda"              : True,
}
```
3. 运行predict.py，输入  
```python
img/street.jpg
```
4. 在predict.py里面进行设置可以进行fps测试和video视频检测。  

## 评估步骤 

1. 本项目使用VOC格式进行评估。  
2. 如果在训练前已经运行过voc_annotation.py文件，代码会自动将数据集划分成训练集、验证集和测试集。如果想要修改测试集的比例，可以修改voc_annotation.py文件下的trainval_percent。trainval_percent用于指定(训练集+验证集)与测试集的比例，默认情况下 (训练集+验证集):测试集 = 9:1。train_percent用于指定(训练集+验证集)中训练集与验证集的比例，默认情况下 训练集:验证集 = 9:1。
3. 利用voc_annotation.py划分测试集后，前往get_map.py文件修改classes_path，classes_path用于指向检测类别所对应的txt，这个txt和训练时的txt一样。评估自己的数据集必须要修改。
4. 在ssd.py里面修改model_path以及classes_path。**model_path指向训练好的权值文件，在logs文件夹里。classes_path指向检测类别所对应的txt。**  
5. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中。

```bash
Author：Huang Jiaqi
Created：2022-05-14
Last updated：2022-05-22
Function：Target detection task for masks using the SSD model for the Masked-Face-Dataset dataset.
由于github上传单个文件不能超过100M，所以我将权重文件和自己训练的模型文件上传到了谷歌云，这是链接：https://drive.google.com/drive/folders/1IECc-7JHDzqFsjBWJm19NuOOk-iloRh-?usp=sharing
```
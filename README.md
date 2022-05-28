# SSD: Training effect of Single-Shot MultiBox Detector target detection model on Masked-Face-Dataset dataset


## Ⅰ. Required environment

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

## Ⅱ. Training steps

###Preparation of the data set

In this paper, we use the VOC format for training, and we need to create our own dataset before training. We use labelimg to label all the image samples in the dataset.

The label files are placed in the VOCdevkit folder under the VOC2007 folder in Annotation before training.  
The image files are placed in JPEGImages under the VOC2007 folder in the VOCdevkit folder before training.   

###Data set processing

After we have finished placing the dataset, we need to use voc_annotation.py to obtain 2007_train.txt and 2007_val.txt for training.   
Modify the parameters inside voc_annotation.py. For the first training you can modify only the classes_path, which is used to point to the txt corresponding to the detection category.   
When training your own dataset, you can create a cls_classes.txt with the categories you need to distinguish.

Translated with www.DeepL.com/Translator (free version)

model_data/cls_classes.txt This could be amended to read：      
```python
mask
ummask
```

Modify classes_path in voc_annotation.py so that it corresponds to cls_classes.txt, and run voc_annotation.py.  

###Start network training

The configuration information for the training parameters is in train.py, the most important part of which is still classes_path in train.py.  
This txt is the same as the one in voc_annotation.py and must be modified to train your own dataset.  
After modifying classes_path you can run train.py and start training. After training multiple epochs, the weights will be generated in the logs folder.  

###Prediction of training results

Two files are used for training result prediction, ssd.py and predict.py. Inside ssd.py, modify model_path and classes_path.  
model_path points to the trained weights file, which is in the logs folder.  
classes_path points to the txt corresponding to the detection category.  
Once you have made the changes you can run predict.py for detection. After running, enter the image path to detect.  

## Prediction steps
### Use your own trained weights
1. Follow the training steps.  
2. In the ssd.py file, change model_path and classes_path to correspond to the trained files in the following section; **model_path corresponds to the weights file under the logs folder, and classes_path is the class into which model_path corresponds**.  
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
3. run predict.py，Input:  
```python
img/street.jpg
```
4. Settings inside predict.py allow fps testing and video video detection.  

## Assessment steps 

1. This project uses the VOC format for evaluation.  
2. If the voc_annotation.py file has been run before training, the code will automatically divide the dataset into a training set, a validation set and a test set. If you want to modify the ratio of the test set, you can modify the trainval_percent under the voc_annotation.py file. trainval_percent is used to specify the ratio of the (training set + validation set) to the test set, by default (training set + validation set):test set = 9:1. train_percent is used to Specify the ratio of the training set to the validation set in (training set + validation set), by default training set:validation set = 9:1.
3. After dividing the test set using voc_annotation.py, go to the get_map.py file and modify the classes_path, which is used to point to the txt corresponding to the test category, which is the same as the txt used for training. The evaluation of your own dataset has to be modified.
4. Modify model_path and classes_path in ssd.py. **model_path points to the trained weights file, in the logs folder. classes_path points to the txt corresponding to the detection category.**  
5. Run get_map.py to get the evaluation results, which will be saved in the map_out folder.


```bash
Author：Huang Jiaqi
Created：2022-05-14
Last updated：2022-05-22
Function：Target detection task for masks using the SSD model for the Masked-Face-Dataset dataset.

Since a single file cannot exceed 100M on github, I uploaded the weights and my trained model files to Google Cloud, here is the link: https://drive.google.com/drive/folders/1IECc-7JHDzqFsjBWJm19NuOOk-iloRh-?usp =sharing
```
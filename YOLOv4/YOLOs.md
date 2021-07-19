# Appendix
### object detector
<img align="center" width="710" alt="Screen Shot 2021-07-13 at 8 32 22 AM" src="https://user-images.githubusercontent.com/6396598/125368357-0658ca00-e3b5-11eb-9ed4-0e568fef0753.png">

## YOLOv1 (https://pjreddie.com/darknet/yolo/)

<img align="center" src = "https://user-images.githubusercontent.com/6396598/125714407-e511b81c-e4b3-4674-874e-e2a17c07da55.png" width="50%" height="50%">

**The input image is divided into an S×S grid (S=7).** If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

**Each grid cell predicts B bounding boxes (B=2) and confidence scores for those boxes.** These confidence scores reflect how confident the model is that the box contains an object, i.e. any objects in the box, P(Objects).

**Each bounding box consists of 5 predictions: x, y, w, h, and confidence.**

**Each grid cell also predicts conditional class probabilities, P(Classi|Object).**


### Network Architecture

<img align="center" src = "https://user-images.githubusercontent.com/6396598/126087866-bee2f522-c88c-4f53-9cbd-ebf385731285.png" width="100%" height="100%">

### Loss function

<img align="center" src = "https://user-images.githubusercontent.com/6396598/125715565-098686aa-518d-4956-847f-d4ef0f18fc08.png" width="80%" height="80%">

## YOLO9000(https://arxiv.org/abs/1612.08242)

### Main contribution
1. Beter
 - Use BatchNormalization
 - High Resolution Classifier
 - Convolutional with Ancher boxs
 - Dimension Cluster
 - Direct Location Prediction
 - Fine-Grained Features
 - Multi-Scale Training
 
2. Faster
 - Darknet
 ![image](https://user-images.githubusercontent.com/6396598/126088599-b55f496c-00c3-4d19-a8ba-368a9466489f.png)

3. Stronger
 - Hierarchical Classification
 - Dataset Combination with WordTree
 - Joint Classification and Detection
 
## YOLOv3

### Main contribution

![image](https://user-images.githubusercontent.com/6396598/125713432-aa61da4c-975e-4eb4-95cc-7f42db372f0d.png)

1. Bounding Box Prediction
 
<img align="center" src = "https://user-images.githubusercontent.com/6396598/125712644-2e59f0bf-a024-4697-bcf1-ce4db9d857d7.png" width="30%" height="30%">

2. Darknet-53

<img align="center" src = "https://user-images.githubusercontent.com/6396598/125712738-d6fb3810-d42b-4035-9c1c-be72c70d4949.png" width="30%" height="30%">

### YOLO series

YOLOv1: https://www.youtube.com/watch?v=eTDcoeqj1_w

YOLO9000: https://www.youtube.com/watch?v=6fdclSGgeio

YOLOv3: https://www.youtube.com/watch?v=HMgcvgRrDcA

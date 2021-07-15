# YOLOv1 (https://pjreddie.com/darknet/yolo/)

<img src = "https://user-images.githubusercontent.com/6396598/125714407-e511b81c-e4b3-4674-874e-e2a17c07da55.png" width="50%" height="50%">

**The input image is divided into an S×S grid (S=7).** If the center of an object falls into a grid cell, that grid cell is responsible for detecting that object.

**Each grid cell predicts B bounding boxes (B=2) and confidence scores for those boxes.** These confidence scores reflect how confident the model is that the box contains an object, i.e. any objects in the box, P(Objects).

**Each bounding box consists of 5 predictions: x, y, w, h, and confidence.**

**Each grid cell also predicts conditional class probabilities, P(Classi|Object).**

### OUTPUT

<img src = "https://user-images.githubusercontent.com/6396598/125714787-c48a1217-c2cb-4c45-b8cb-1973b90c72b1.png" width="50%" height="50%">

### Network Architecture

<img src = "https://user-images.githubusercontent.com/6396598/125715441-cf35cae7-865f-4c60-99fd-19ab750c2e39.png" width="80%" height="80%">

### Loss function

<img src = "https://user-images.githubusercontent.com/6396598/125715565-098686aa-518d-4956-847f-d4ef0f18fc08.png" width="80%" height="80%">

# YOLOv3

### Main contribution
![image](https://user-images.githubusercontent.com/6396598/125713432-aa61da4c-975e-4eb4-95cc-7f42db372f0d.png)

1. Bounding Box Prediction
 
<img src = "https://user-images.githubusercontent.com/6396598/125712644-2e59f0bf-a024-4697-bcf1-ce4db9d857d7.png" width="50%" height="50%">

2. Darknet-53

<img src = "https://user-images.githubusercontent.com/6396598/125712738-d6fb3810-d42b-4035-9c1c-be72c70d4949.png" width="30%" height="30%">

# YOLOv4
1. Performance of YOLOv4
<img width="621" alt="image" src="https://user-images.githubusercontent.com/6396598/125368520-56d02780-e3b5-11eb-9188-18861b026389.png">
* Twice faster than EfficientDet
* 
## Main contribution

1. Utilize SOTA Bag-of-Freebies and Bag-of-Specials methods during detector training
2. 
* Back of Freebies (only change the training strategy or only  increase the training cost method)

  - Data augmentation
  -- Mosaic augmentation
  -- Self adversarial training
  - Regularization
  - Loss function
* Bag of specials

3. Modify SOTA methods(CBN, PAN, SAM, etc) and make them to be available sing GPU training


## Appendix
### object detector
<img width="710" alt="Screen Shot 2021-07-13 at 8 32 22 AM" src="https://user-images.githubusercontent.com/6396598/125368357-0658ca00-e3b5-11eb-9ed4-0e568fef0753.png">


### YOLO series
YOLOv1: https://www.youtube.com/watch?v=eTDcoeqj1_w


YOLO9000: https://www.youtube.com/watch?v=6fdclSGgeio


YOLOv3: https://www.youtube.com/watch?v=HMgcvgRrDcA


### reference
YOLOv4: https://arxiv.org/pdf/2004.10934.pdf

ScaledYOLOv4: https://arxiv.org/abs/2011.08036

Github: https://github.com/AlexeyAB/darknet

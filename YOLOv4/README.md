# YOLOv4

### Overview

1. Performance: 2 times faster than EfficientDet, 10% AP and 12% fps higher than YOLOv3
 <img aligh="center" src="https://user-images.githubusercontent.com/6396598/125368520-56d02780-e3b5-11eb-9188-18861b026389.png" width="40%" height="40%">
  
2. Network Architecture:YOLOv4 = CSPDarknet53 backbone + SPP additional module + PANet path-aggregation neck + YOLOv3 head
  <img aligh="center" src="https://user-images.githubusercontent.com/6396598/125764022-a4a43460-0bd9-4f3b-9369-786564a20d2d.png" width="80%" height="80%">

   

3. Utilize SOTA Bag-of-Freebies and Bag-of-Specials methods during the detector training

4. Modify SOTA methods and make them more effecient and suitabble for single GPU training

### Main contribution

#### 1. Utilize SOTA Bag-of-Freebies and Bag-of-Specials methods during detector training

**Back of Freebies (only change the training strategy or only  increase the training cost method)**
<img align="center" src = "https://user-images.githubusercontent.com/6396598/125727259-644d3cbc-3e5f-4748-a298-36664a575d2a.png" width="50%" height="50%">

 - Data augmentation
 
   <img align="center" src = "https://user-images.githubusercontent.com/6396598/125741393-102239e1-e021-4e14-b11d-cfe62e41093f.png" width="50%" height="50%">

   - Random erase and CutOut: randomly select the rectangle regions in an image and fill in a random value or zero.
   - MixUp: multiply 2 images and superimpose with differenct coefficient ratios
   - CutMix:cover the cropped image to rectangle region of other images
   - Styletransfer GAN
   - Mosaic: mixes 4 training images <img align="right" src = "https://user-images.githubusercontent.com/6396598/125745632-86f902b8-c129-4cb0-bead-886f7c1129f6.png" width="50%" height="50%">
   - Self adversarial training
 - Objective function of BBox regression
   - MSE (Meab Sqared Error)
   - IoU
   - GIoU: Consider the shape and orientation of objects <img align="right" src = "https://user-images.githubusercontent.com/6396598/126089976-7d5c6830-ed6f-418b-9a55-5abea8f4a8df.png" width="50%" height="50%">
   - DIoU: Additionly consider the distance of the center of an object <img align="right" src = "https://user-images.githubusercontent.com/6396598/126091728-9634d2c0-cf42-491c-be03-39e5c1a36989.png" width="50%" height="50%">
   - CIoU: Simultaneously considers the overlapping area, the distance between center points, and the aspect ratio.
 - Regularization
   - DropOut:
   - DropPath: <img align="right" src = "https://user-images.githubusercontent.com/6396598/126093821-689ab269-ceed-4ab6-a6b3-0b8f3899623a.png" width="50%" height="50%">

   - Spatial DropOut:
   - DropBlock: <img align="right" src = "https://user-images.githubusercontent.com/6396598/126093042-811fd4aa-c9e6-48b1-892e-d5f88779d7c1.png" width="50%" height="50%">

   
**Bag of specials**

<img align="center" src = "https://user-images.githubusercontent.com/6396598/125727338-2399db94-db02-41e5-8f28-8bc022220224.png" width="50%" height="50%">

### 2. Modify SOTA methods(CBN, PAN, SAM, etc) and make them to be available sing GPU training
 - CBN (Cross Iteration Batch Normalization)
 
 - PAN (Path Aggregation Network)
 <img align="center" src = "https://user-images.githubusercontent.com/6396598/125761460-e482de59-e2df-48e0-87c5-21ef7c443ed3.png" width="50%" height="50%">

 - SAM (Spatial Attention Module)
 <img align="center" src = "https://user-images.githubusercontent.com/6396598/125761346-03bd2fab-4d56-4523-ad3e-bf94dc4aad85.png" width="50%" height="50%">

### 3. Architecture
 - Backbone: CSPDarknet53
  - Bag of Freebies for backbone: CutMix, Mosaic data augmentation, DropBlock, Class label smoothing
  - Bag of Specials for backbone: Mish activation, Cross-stage partial connections (CSP), Multi-input weighted residual conncections (MiWRC)
  
 - Neck: SPP, PAN
 - Head: YOLOv3
 

### reference

- YOLOv4: https://arxiv.org/pdf/2004.10934.pdf
         https://www.youtube.com/watch?v=_JzOFWx1vZg   

- ScaledYOLOv4: https://arxiv.org/abs/2011.08036
- Github: https://github.com/AlexeyAB/darknet
- Hoya012's blog: https://hoya012.github.io/blog/yolov4/



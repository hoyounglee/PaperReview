# Object Detection Overview

## Object Detection
### Definition
> Object detection is a computer vision technique that allows us to identify and locate objects in an image or video. 
> With this kind of identification and localization, object detection can be used to count objects in a scene
> and determine and track their precise locations, all while accurately labeling them.

### Difference from Classification
> Classification task looks for the 'class' of the target scene. The model can only output a **single** class name from a single image.
> However, object detection can find the location of the objects along with the class name of them. Besides, object detector can find **multiple** object's locations and class
> from a single image (multi object detection)

## Overall process of Object Detection
- single stage detector vs two stage detector
<img width="690" alt="image" src="https://user-images.githubusercontent.com/32179857/176093460-0edca585-a9bb-4d08-8b86-210878fa8019.png">

(Image from https://stackoverflow.com/questions/65942471/one-stage-vs-two-stage-object-detection)

## Terms
#### RoI
- Region of Interest
<img width="600" alt="image" src="https://user-images.githubusercontent.com/32179857/176093586-32bc6bb9-44df-46bf-baf0-5916a254e0b3.png">

(Image from https://kr.mathworks.com/help/visionhdl/ref/visionhdl.roiselector-system-object.html)

#### IoU
- Intersection over Union
<img width="541" alt="image" src="https://user-images.githubusercontent.com/32179857/176093902-b7915d25-59a4-4695-bb96-d92f7602dc8b.png">

(Image from https://www.researchgate.net/figure/Intersection-over-Union-IOU-calculation-diagram_fig2_335876570) 

#### NMS
- Non maximum suppression
```python
   import torch
   from IoU import intersection_over_union

   def nms(bboxes, iou_threshold, threshold, box_format='corners'):
       # bboxes가 list인지 확인합니다.
       assert type(bboxes) == list

       # box 점수가 threshold보다 높은 것을 선별합니다.
       # box shape는 [class, score, x1, y1, x2, y2] 입니다.
       bboxes = [box for box in bboxes if box[1] > threshold]
       # 점수 오름차순으로 정렬합니다.
       bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
       bboxes_after_nmn = []

       # bboxes가 모두 제거될때 까지 반복합니다.
       while bboxes:
           # 0번째 index가 가장 높은 점수를 갖고있는 box입니다. 이것을 선택하고 bboxes에서 제거합니다.
           chosen_box = bboxes.pop(0)

           # box가 선택된 box와의 iou가 임계치보다 낮거나
           # class가 다르다면 bboxes에 남기고, 그 이외는 다 없앱니다.
           bboxes = [box for box in bboxes if box[0] != chosen_box[0] \
                  or intersection_over_union(torch.tensor(chosen_box[2:]),
                                             torch.tensor(box[2:]),
                                             box_format=box_format)
                       < iou_threshold]

           # 선택된 박스를 추가합니다.
           bboxes_after_nmn.append(chosen_box)

       return bboxes_after_nmn
```


#### mAP
- Mean Average Precision
- TP, FP, TN, FN

    <img width="680" alt="image" src="https://user-images.githubusercontent.com/32179857/176380654-d8c7b29c-539b-4c64-895b-c26dd7b9005f.png">

- Suppose we detect 10 apples for the image that has 5 apples.
- We write the result in the order of confidence scoer (highly confident)
<img width="632" alt="image" src="https://user-images.githubusercontent.com/32179857/176098792-085bb2f3-56a9-4e46-88d8-e10a6a9f8ae0.png">

- AP (Average Precision)
<img width="629" alt="image" src="https://user-images.githubusercontent.com/32179857/176098840-bbd6cad3-eab8-40c4-80d8-05011de64169.png">
<img width="629" alt="image" src="https://user-images.githubusercontent.com/32179857/176098994-40ef704f-5557-4ac6-b192-de20f652e091.png">

- mAP: AP over all classes


## 1-Stage Detectors
- backbone + detector head

## 2-Stage Detectors
- backbone + rpn neck + detector headd
- 

## Backbone, Neck, Head


## Transformers

   
### References
- https://deepsense.io/region-of-interest-pooling-explained/
- 

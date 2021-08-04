## Meta Pseudo Labels
- paper: [Archive link](https://arxiv.org/pdf/2003.10580.pdf)

### Introduction
- The paper is upgraded version of 'Pseudo labeling'
- Despite the strong performance of Pseudo Labels methods, they have a drawback
    - if pseudo labels are inaccurate --> the student will learn from inaccurate data
- The authors design a systematic mechanism for the teacher to correct the bias by observing how its pseudo labels would affect the student
    - The model utilizes the feedback from student during training
    - The feedback signal is used as a reward to train the teacher
    - parallel training
        - 1) the student learns from a minibatch of psedo labeled data annotated by the teacher
        - 2) the teacher learns from the reward signal of how the student performs on a minibatch
    - The performance of the model(accuracy) has been increased about 2% from SOTA (88.5 -> 90.2, ImageNet)
   
### Related works
- Semi-Supervised Learning (SSL)
    - In SSL, a typical solution first employs an existing model qξ (trained on limited labeled data) to predict the class for each data point from an unlabeled set 
      and utilizes the prediction to construct the target distribution which can be done in two ways:
      - Hard label : q∗(Y|x) = one-hot (argmax (qξ(y|x))
      - Soft label : q∗(Y|x) = qξ(Y|x)
- Knowledge distillation
    - For each data point x, the predicted distribution of the large model q_large(x) is directly taken as the target distribution, i.e. q∗(Y|x) = q_large(Y|x)
- Label Smoothing

    ![image](https://user-images.githubusercontent.com/32179857/127939284-5482c40b-b638-4788-87b4-582fa5725b9c.png)
- Temperature Scaling

    ![image](https://user-images.githubusercontent.com/32179857/127939315-3cbaf1a5-200d-4da5-895b-e29992437f9c.png)





### Methods
   ![image](https://user-images.githubusercontent.com/32179857/127819464-3696053e-ee57-4c02-bd34-88800c17eaae.png)
- The main difference is that in Meta Pseudo Labels, the teacher receives feedback of the student's performance on a labeled dataset.
- below is the original Pseudo Labels(PL) loss function
    <p align="left">
      <img width="500" height="100" src="https://user-images.githubusercontent.com/32179857/127825492-44853ee8-a5e0-48fc-9c3c-7910cc9209b7.png">
    </p>
- we can notice that the optimal student parameter ![image](https://user-images.githubusercontent.com/32179857/127825852-98a89cbc-a4b7-47af-897a-51738057f112.png) is always depend on the teacher parameter ![image](https://user-images.githubusercontent.com/32179857/127825912-01c755da-653e-478b-a633-77cc0dc3de90.png)
- so we can futher optimze the loss ![image](https://user-images.githubusercontent.com/32179857/127826107-ff33e1df-b835-4b04-abbd-04996b20639b.png) with repspect to ![image](https://user-images.githubusercontent.com/32179857/127826146-3a56528c-5199-43a5-9c27-44e5657b710c.png)

  <p align="left">
      <img width="500" height="100" src="https://user-images.githubusercontent.com/32179857/127826176-bd8ae76c-7c09-442c-9f17-42d55f368c9c.png">
  </p>
- since we connected student's feedback with the teacher, by optimizing the teacher's parameter according to the performance of the student on labeled data, the pseudo labels can be adjusted accordingly to further improve student's performance.
- when we see by gradient update with the calculus,
    - Student:
    
        ![image](https://user-images.githubusercontent.com/32179857/127826699-37f8c6a1-286d-4eee-a2d5-8801edbacf50.png)
    - Teacher:
    
        ![image](https://user-images.githubusercontent.com/32179857/127826722-3c021ce8-0a6b-431d-b1c6-c834f96d93df.png)

#### Training procedures
<p align="center">
    <img width="500" height="300" src="https://user-images.githubusercontent.com/32179857/127942122-590fffba-9371-44fe-8df7-1c991a90cc2d.png">
</p>
<p align="center">
    <img width="500" height="400" src="https://user-images.githubusercontent.com/32179857/127942134-d2bae3f0-048d-4cc3-adb1-5dfc9338e0d6.png">
</p>




### Experiments
- MPL is tested in two scenarios
    - where limited data is available
    - where the full labeled dataset is used

<p align="center">
    <img width="500" height="800" src="https://user-images.githubusercontent.com/32179857/127942265-149a3be3-eed5-4cbd-862d-e89d87ad231e.png">
</p>

### Analysis
- MPL is not label correction

Since the teacher in MPL provides the target distribution for the student to learn and observes the student’s performance to improve itself, it is intuitive to think that the teacher tries to guess the correct labels for the student. To prove this, the authors plotted the training accuracy of all three: a purely supervised model, the teacher model, and the student model
<p align="center">
    <img width="500" height="400" src="https://user-images.githubusercontent.com/32179857/127942657-bb8a209c-d09b-461d-8b3b-449259439dd7.png">
</p>

As shown, the training accuracy of both the teacher and the student stays relatively low. Meanwhile, the training accuracy of the supervised model eventually reaches 100% much earlier. If MPL was simply performing label correction, then these accuracies should have been high. Instead, it looks like the teacher in MPL is trying to regularize the student to prevent overfitting which is crucial when you have a limited labeled dataset.

- MPL is not only a regularization Strategy

As the authors said above that the teacher model in MPL is trying to regularize the student, it is easy to think that the teacher might be injecting noise so that the student doesn’t overfit. Turns out it isn’t the case. If the teacher has to inject noise in students’ training, it can do it in two ways: 

    1.) By flipping the target class, eg. tell the student that the image of a car is an image of a horse.
    2.) By dampening the target distribution. 
To confirm this, the authors visualized a few target distributions predicted by the teacher model in ReducedMPL for images in the TinyImages dataset.

<p align="center">
    <img width="500" height="600" src="https://user-images.githubusercontent.com/32179857/127942989-c2e28dee-203b-4f02-84c9-3b8e654a0a69.png">
</p>


### Code
- refered by https://github.com/kekmodel/MPL-pytorch
![image](https://user-images.githubusercontent.com/32179857/128143267-1b14f7ba-4d9e-49a7-a4d5-5e90cfac3b48.png)



### References
- https://github.com/kekmodel/MPL-pytorch
- https://medium.com/@nainaakash012/meta-pseudo-labels-6480acb1b68

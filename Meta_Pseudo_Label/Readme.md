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
   
### Methods
   ![image](https://user-images.githubusercontent.com/32179857/127819464-3696053e-ee57-4c02-bd34-88800c17eaae.png)

### Experiments

#### Small scale experiments

#### Large scale experiments

### Conclusion

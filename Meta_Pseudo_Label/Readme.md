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
- The main difference is that in Meta Pseudo Labels, the teacher receives feedback of the student's performance on a labeled dataset.
- below is the original Pseudo Labels(PL) loss function
    <p align="left">
      <img width="500" height="100" src="https://user-images.githubusercontent.com/32179857/127825492-44853ee8-a5e0-48fc-9c3c-7910cc9209b7.png">
    </p>
- we can notice that the optimal student parameter ![image](https://user-images.githubusercontent.com/32179857/127825852-98a89cbc-a4b7-47af-897a-51738057f112.png) is always depend on the teacher parameter ![image](https://user-images.githubusercontent.com/32179857/127825912-01c755da-653e-478b-a633-77cc0dc3de90.png)
- so we can futher optimze the loss ![image](https://user-images.githubusercontent.com/32179857/127826107-ff33e1df-b835-4b04-abbd-04996b20639b.png) with repspect to ![image](https://user-images.githubusercontent.com/32179857/127826146-3a56528c-5199-43a5-9c27-44e5657b710c.png)
![image](https://user-images.githubusercontent.com/32179857/127826176-bd8ae76c-7c09-442c-9f17-42d55f368c9c.png)
- since we connected student's feedback with the teacher, by optimizing the teacher's parameter according to the performance of the student on labeled data, the pseudo labels can be adjusted accordingly to further improve student's performance.
- when we see by gradient update with the calculus,
    - Student:
        ![image](https://user-images.githubusercontent.com/32179857/127826699-37f8c6a1-286d-4eee-a2d5-8801edbacf50.png)
    - Teacher:
        ![image](https://user-images.githubusercontent.com/32179857/127826722-3c021ce8-0a6b-431d-b1c6-c834f96d93df.png)




### Experiments

#### Small scale experiments

#### Large scale experiments

### Conclusion

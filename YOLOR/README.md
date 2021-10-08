# YOLOR(You Only Learn One Representation: Unified Networks for Multiple tasks) [paper](https://arxiv.org/pdf/2105.04206.pdf)

## Overview

This paper proposed unified network to integrate ***implicit knowledge(?)*** and ***explicit knowledge(?)***, and enable the learned model to contain a ***general representation***, and this general representation enable sub-representations ***suitable for various tasks***

![image](https://user-images.githubusercontent.com/6396598/128137820-e900dd05-5d86-4d98-aa31-cf268ecf79cd.png)

- **In General,**
  - explicit knowledge: the features obtained from the shallow layers
  - implicit knowledge: the features obtained from the deep layers
- **In this paper,**
  - explicit knowledge: Knowledge that directly directly correspond to observation
  - implicit knowledge: knowledge that is implicit in the model and has nothing to do with observation.   

## Related work

-------------------------------------------
### 1. Explicit deep learning
**Methods that can automatically adjust or select features based on input data**

Transformer, Non-local networks, etc

-------------------------------------------
### 2. Implicit deep learning
Implicit neural representations, deep equilibrium models
  
* Appendix: Deep equaliblrim models [ref](http://implicit-layers-tutorial.org/deep_equilibrium_models/)
  
  ![image](https://user-images.githubusercontent.com/6396598/128139581-b1847f7f-e8d9-447d-8851-a3d16ba5e71e.png)

    implicit knowldege(?): something irrelevant to input (or observation), but learned inside model
 
-------------------------------------------
### 3. Knowledge modeling
**Integrate implicit knowledge and explicit knowledge**

- sparse representation: exampler, predefined over complete, or learned dictionary to perform modeling
- memory networks: combining various forms of embedding to form memory, and enable memory to be dynamically added or changed
memory networks

## How implicit knowledge works?
...

## Implicit knowledge in unified networks
-------------------------------------------
### Formulation of implicit knowledge
**1. Conventional networks**
![image](https://user-images.githubusercontent.com/6396598/128148020-2879bf0d-afe8-4fcf-a6b2-bb2640ef048f.png)

**2. Unified networks**
![image](https://user-images.githubusercontent.com/6396598/128151354-7dcbdbde-d8d0-437b-9590-cb69d4418557.png)

![image](https://user-images.githubusercontent.com/6396598/128154517-9f3f918b-6291-4236-bcc4-28844a9ad3cd.png)
 
 ### Modeling implicit knowledge
 ![image](https://user-images.githubusercontent.com/6396598/128267666-738ee006-ef35-4498-9f64-c5a08be0415a.png)

 **1. vector/Matrix/Tensor: Use vector <a href="https://www.codecogs.com/eqnedit.php?latex=z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z" title="z" /></a> directly**
 
 **2. Neural Network: <a href="https://www.codecogs.com/eqnedit.php?latex=Wz" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Wz" title="Wz" /></a> Use vector <a href="https://www.codecogs.com/eqnedit.php?latex=z" target="_blank"><img src="https://latex.codecogs.com/gif.latex?z" title="z" /></a> as the prior of implicit knowledge, the use the weight matrix <a href="https://www.codecogs.com/eqnedit.php?latex=W" target="_blank"><img src="https://latex.codecogs.com/gif.latex?W" title="W" /></a>**
 
 **3. Matrix Factorization: <img src="https://latex.codecogs.com/gif.latex?Z^{T}c" title="Z^{T}c" />**
### Implementation
**1. Implicit_add**
```
class ImplicitA(nn.Module):
    def __init__(self, channel):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit
```
**2. Implicit_mul**
```
class ImplicitM(nn.Module):
    def __init__(self, channel):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=1., std=.02)

    def forward(self):
        return self.implicit
```

**3. Implicit_concat**
```
class ImplicitC(nn.Module):
    def __init__(self, channel):
        super(ImplicitC, self).__init__()
        self.channel = channel
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, std=.02)

    def forward(self):
        return self.implicit
```
-------------------------------------------
### Training
![image](https://user-images.githubusercontent.com/6396598/128272323-ce35c72b-2628-4529-8861-e8432327eb55.png)

-----------------------------------------
### Inference
![image](https://user-images.githubusercontent.com/6396598/128270841-280e9792-6e92-448d-bdca-7eaffe83ed8f.png)
## Experiment
![image](https://user-images.githubusercontent.com/6396598/128281845-c1bf32d5-68fe-4b42-8628-8dda5eb6b8d4.png)
### 1. Feature alignment for FPN
![image](https://user-images.githubusercontent.com/6396598/128271536-703543c2-a3bc-42a2-9bf3-c6ac7782485c.png)
### 2. Prediction refinement for object detection
![image](https://user-images.githubusercontent.com/6396598/128271609-15a236d9-b3cb-482e-9ba9-a0af55d4fa9a.png)
### 3. Canoical representation for multi-task
![image](https://user-images.githubusercontent.com/6396598/128272539-652ab4a6-38c2-4bc5-afe8-26fabfd09a4e.png)
### 4. Implicit modeling with different operator
![image](https://user-images.githubusercontent.com/6396598/128272663-1d7c1079-ea4b-4a2c-8e75-31695c423917.png)
### 5. Modeling implicit knowledge in different ways
![image](https://user-images.githubusercontent.com/6396598/128273615-ad267517-1171-4432-a359-8e164c875471.png)
### Analysis of implicit models
![image](https://user-images.githubusercontent.com/6396598/128273690-09e11ec8-e429-4591-9559-681bbd3a4170.png)
### Implicit knowledge for object detection
![image](https://user-images.githubusercontent.com/6396598/128273729-4539db0e-701d-41d3-98f9-99883fd38fe2.png)
![image](https://user-images.githubusercontent.com/6396598/128282117-718a0e8a-72df-4ca3-8f25-806f01cdab50.png)

## Contributions
1. A unified ntworks that can accomplish various tasks by intergrating implicit knowledge and explicit knowledge
2. Introducing *Kernel space alignment*, *prediction refinement*, *multi-task learning into the implicit knowledge learning process*
3. Discussing the ways of using *vector*, *neural nework*, or *matrix factorization* as a tool to model implicit knowledge
4. Combined with SOTA methods( achived comparable accuracvy as Scaled -YOLOv4-P7 and the inference speed has been increased 88%)

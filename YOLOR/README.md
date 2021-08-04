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
### 1. Explicit deep learning
**Methods that can automatically adjust or select features based on input data**

Transformer, Non-local networks, etc

### 2. Implicit deep learning

Implicit neural representations, deep equilibrium models
  
* Appendix: Deep equaliblrim models [ref](http://implicit-layers-tutorial.org/deep_equilibrium_models/)
  
  ![image](https://user-images.githubusercontent.com/6396598/128139581-b1847f7f-e8d9-447d-8851-a3d16ba5e71e.png)

    implicit knowldege(?): something irrelevant to input (or observation), but learned inside model
 
### 3. Knowledge modeling
**Integrate implicit knowledge and explicit knowledge**

- sparse representation: exampler, predefined over complete, or learned dictionary to perform modeling
- memory networks: combining various forms of embedding to form memory, and enable memory to be dynamically added or changed
memory networks

## How implicit knowledge works?
...

## Implicit knowledge in unified networks
**1. Conventional networks**
![image](https://user-images.githubusercontent.com/6396598/128148020-2879bf0d-afe8-4fcf-a6b2-bb2640ef048f.png)

**2. Unified networks**
![image](https://user-images.githubusercontent.com/6396598/128151354-7dcbdbde-d8d0-437b-9590-cb69d4418557.png)

 ![image](https://user-images.githubusercontent.com/6396598/128154517-9f3f918b-6291-4236-bcc4-28844a9ad3cd.png)
## Reference


## Contributions
1. A unified ntworks that can accomplish various tasks by intergrating implicit knowledge and explicit knowledge
2. Introducing *Kernel space alignment*, *prediction refinement*, *multi-task learning into the implicit knowledge learning process*
3. Discussing the ways of using *vector*, *neural nework*, or *matrix factorization* as a tool to model implicit knowledge
4. Combined with SOTA methods( achived comparable accuracvy as Scaled -YOLOv4-P7 and the inference speed has been increased 88%)

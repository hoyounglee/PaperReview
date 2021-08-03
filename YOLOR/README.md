# YOLOR(You Only Learn One Representation: Unified Networks for Multiple tasks) [paper](https://arxiv.org/pdf/2105.04206.pdf)

## Overview

Proposed a unified network to integrate ***implicit knowledge(?)*** and ***explicit knowledge(?)***, and enable the learned model to contain a ***general representation***, and this general representation enable sub-representations ***suitable for various tasks***

![image](https://user-images.githubusercontent.com/6396598/127944518-946439ff-3761-4b30-8c2e-5cc13a1a70c5.png)

## Contributions
1. A unified ntworks that can accomplish various tasks by intergrating implicit knowledge and explicit knowledge
2. Introducing *Kernel space alignment*, *prediction refinement*, *multi-task learning into the implicit knowledge learning process*
3. Discussing the ways of using *vector*, *neural nework*, or *matrix factorization* as a tool to model implicit knowledge
4. Combined with SOTA methods( achived comparable accuracvy as Scaled -YOLOv4-P7 and the inference speed has been increased 88%)

## Background
### 1. Explicit deep learning
**Methods that can automatically adjust or select features based on input data**
Transformer, Non-local networks, ...

### 2. Implicit deep learning
**Implicit deep knowledge learning and implicit differential derivative**
Implicit neural representations, deep equilibrium models

### 3. Knowledge modeling
**Integrate implicit knowledge and explicit knowledge**
sparse representation, memory networks

## Implicit knowledge
Object function of conventional network training
![image](https://user-images.githubusercontent.com/6396598/127955988-c4bf206e-6c30-4348-8a1f-7dfc853634f1.png)


## Reference

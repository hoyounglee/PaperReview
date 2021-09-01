# EGO-TOPO: Environment Affordances from Egocentric Video [[paper](https://arxiv.org/abs/2001.04583)]

## Background
### Scene understanding [[ref](https://alexgkendall.com/media/presentations/2019_icvss_sicily_kendall.pdf)]
“Scene understanding is to analyze a scene by considering the geometric and semantic context of its contents and the intrinsic relationships between them.” [Indoor Scene Understanding in 2.5/3D: A Survey. Naseer et al. 2018.]

"Scene understanding is to extract a minimal representation of the world which can be used to evaluate action."
* How to? ==> Learn semantics, motion and geometry.
![image](https://user-images.githubusercontent.com/6396598/131304545-26ef3b32-2c9a-422a-9c06-01ab0a31833a.png)

## Overview

In this paper: Scene understanding for **ego-centric video** --> ***make a zone-affortance graph and utilize it for anticipating future actions***
 - scene understanding is largely about answering the who/where/what questions of recognition: what objects are present? is it an indoor/outdoor scene? where is the person and what are they doing?
 - "Given an egocentric video, we build a topological map of the environment that reveals activity-centric zones and the sequence in which they are visited."
 - converts egocentric video into a topological map consisting of activity “zones” and their rough spatial proximity.
![image](https://user-images.githubusercontent.com/6396598/130926088-1718cda0-7ce5-42ae-9235-523b11a39cf4.png)

## Main idea
1. train a **zone localization network** to discover commonly visited spaces from egocentric video.
2. Then, given a novel video, assign video clips to zones and **create a topological map (graph)** for the environment.
3. We further link zones based on their function across video instances to create consolidated maps.
4. Finally, we leverage the resulting graphs to uncover environment affordances and anticipate future actions in long videos.

### 1. Discovering Activity-Centric Zones
Problems of conventional method to discover zones
1. Visual feature based: Manipulated obbjects often feature prominently in ego-video. (Makeing the features sensitive to the set of objects present).
2. SLAM localization based: often unrelialble due to quick motions characteristic of egocentric video.

Proposed:
![image](https://user-images.githubusercontent.com/6396598/131055736-77f00d54-2b08-4338-82c6-44b3976746f3.png)

1. Sample pairs of frames from videos that are segmented into a series of action clips.
  
  - Assumption: Two training frames are similar if (1) they are near in time (separated by fewer than 15 frames) or from the same action clip, **or** (2) there are at least 10 inlier keypoints consistent with their estimated homography.
  - for (1), Siamese network wit a ResNet-18 backbone, followed by a 5 layer MLP.
  - for (2) Superpoint descriptor[[ref](http://www.cv-learn.com/20201227-cvpr2020-slam-malisiewicz)]
  
 ![image](https://user-images.githubusercontent.com/6396598/131592563-4e927125-4539-4142-a29a-ed3150b927bf.png)

### 2. Creating the Topological Affordance Grap

- per-video 

![image](https://user-images.githubusercontent.com/6396598/131617449-4ef69343-72f6-4336-ac50-7337c6325dea.png)

- cross-video

 For each node <img src="https://render.githubusercontent.com/render/math?math=n_{i}">, they utilize a pretrained action/object classifier and compute a ***node-level functional similarity score***.
 
 <img src="https://user-images.githubusercontent.com/6396598/131455516-be7dd234-2e1f-4a31-ab87-a8cb6b304fd2.png" width="50%" height="50%" align="center">
 <img src="https://user-images.githubusercontent.com/6396598/131456358-c846fcd0-882a-4d95-91b4-4f60b99a1e09.png" width="50%" height="50%" align="center">

### 3. Inferring Environment Affordances
- They utilize the topological graph to predict a zone's affordances -- all likely interactions possible at that zone.
- Learning scene affordances is especially important when **an agent must use a previously unseen environment to perform a task.**
- However, since each clip of an ego-video shows a zone being used only for a single interaction, **it falls short of capturing all likely interactions at that location**.
![image](https://user-images.githubusercontent.com/6396598/131596096-7902cc17-6a00-49df-9fb7-38e2d531b60b.png)

### 4. Anticipating Future Actions in Long Video
 - In the anticipation task, they see a fraction of a long video (e.g., the first 25%), and from that we must predict what actions will be done in the future.
 - For long range anticipation, models need to understand how much progress has been made on the composite activity so far, and anticipate what actions need to be done in the future to complete it. For this, ***a structured representation of all past activity and affordances is essential***.
 ![image](https://user-images.githubusercontent.com/6396598/131610157-c23aadce-3cc6-4186-b340-970994e66ccc.png)

## Experiments
### 1. dataset
- EGTEA Gaze+: 32subjects folling 7 recipes in a single kitchen. 53 objects and 19 actions.
- EPIC_Kitchens: 352 objects and 125 actions.

### 2. EGO-TOPO for Environment Affordances
![image](https://user-images.githubusercontent.com/6396598/131612264-07059136-a561-4faf-8c9a-60c144b462f1.png)

### 3. EGO-TOPO for Long Term Action Anticipation
![image](https://user-images.githubusercontent.com/6396598/131612640-56326de0-a782-4506-9a2b-d2b0da4e2bb8.png)

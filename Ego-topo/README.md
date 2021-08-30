# EGO-TOPO: Environment Affordances from Egocentric Video [[paper](https://arxiv.org/abs/2001.04583)]

## Background
### Scene understanding [[ref](https://alexgkendall.com/media/presentations/2019_icvss_sicily_kendall.pdf)]
“Scene understanding is to analyze a scene by considering the geometric and semantic context of its contents and the intrinsic relationships between them.” [Indoor Scene Understanding in 2.5/3D: A Survey. Naseer et al. 2018.]

"Scene understanding is to extract a minimal representation of the world which can be used to evaluate action."
* How to? ==> Learn semantics, motion and geometry.
![image](https://user-images.githubusercontent.com/6396598/131304545-26ef3b32-2c9a-422a-9c06-01ab0a31833a.png)

## Overview

In this paper: Scene understanding for **ego-centric video**.
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
1. Visual feature based:
2. SLAM localization based:

Proposed:
1. Sample pairs of frames from videos that are segmented into a series of action clips.
  - Two training frames are similar if 
  
  (1) they are near in time (separated by fewer than 15 frames) or from the same action clip, 
  
  **or**
  
  (2) there are at least 10 inlier keypoints consistent with their estimated homography.
2. 
Localization network
![image](https://user-images.githubusercontent.com/6396598/131055736-77f00d54-2b08-4338-82c6-44b3976746f3.png)



### 2. Creating the Topological Affordance Grap

![image](https://user-images.githubusercontent.com/6396598/131293402-0184bec8-96cf-4925-969a-4cf5b8ce86e3.png)

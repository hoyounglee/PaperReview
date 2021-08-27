# EGO-TOPO: Environment Affordances from Egocentric Video [[paper](https://arxiv.org/abs/2001.04583)]

## Background
### Scene understanding [[ref](https://alexgkendall.com/media/presentations/2019_icvss_sicily_kendall.pdf)]
“Scene understanding is to analyze a scene by considering the geometric and semantic context of its contents and the intrinsic relationships between them.” [Indoor Scene Understanding in 2.5/3D: A Survey. Naseer et al. 2018.]

In this paper,
scene understanding is largely about answering the who/where/what questions of recognition: what objects are present? is it an indoor/outdoor scene? where is the person and what are they doing?

![image](https://user-images.githubusercontent.com/6396598/130902053-cad113a2-e0ad-43f2-af93-863cb9ab1c85.png)
![image](https://user-images.githubusercontent.com/6396598/130902431-f3ba12fa-a6fe-44b9-9181-b8c856f936f1.png)


## Overview

![image](https://user-images.githubusercontent.com/6396598/130926088-1718cda0-7ce5-42ae-9235-523b11a39cf4.png)

main purpose of this paper: converts egocentric video into a topological map consisting of activity “zones” and their rough spatial proximity.

"Given an egocentric video, we build a topological map of the environment that reveals activity-centric zones and the sequence in which they are visited."

## Main idea
1. train a **zone localization network** to discover commonly visited spaces from egocentric video.
2. Then, given a novel video, assign video clips to zones and **create a topological map (graph)** for the environment.
3. We further link zones based on their function across video instances to create consolidated maps.
4. Finally, we leverage the resulting graphs to uncover environment affordances and anticipate future actions in long videos.

### 1. Discovering Activity-Centric Zones
Localization network
![image](https://user-images.githubusercontent.com/6396598/131055736-77f00d54-2b08-4338-82c6-44b3976746f3.png)

Two training frames are similar if 

  (1) they are near in time (separated by fewer than 15 frames) or from the same action clip, **or**
  (2) there are at least 10 inlier keypoints consistent with their estimated homography.

### 2. Creating the Topological Affordance Grap

![image](https://user-images.githubusercontent.com/6396598/131096686-9cabbefd-0c8f-40c6-acec-8ecaed082d7a.png)

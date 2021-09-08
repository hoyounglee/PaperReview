Slowfast Networks for Video Recognition  [[Paper]](https://arxiv.org/pdf/1812.03982.pdf)
===

## Abstract
---


**SlowFast networks for video recognition**   

(i) A Slow pathway, operating at low frame rate, to capture spatial semantics   
(ii) A Fast pathway, operating at high frame rate, to capture motion at fine temporal resolution.  The Fast pathway can be made very lightweight by reducing its channel capacity, yet can learn useful temporal information for video recognition.   

Our models achieve strong performance **for both action classification and detection** in video.    
<br/>

## Introduction
---

The categorical spatial semantics of the visual content often evolve slowly.   
On the other hand, the motion being performed can evolve much faster than their subject identities, such as clapping, waving, shaking, walking, or jumping.   

Based on this intuition, we present a two-pathway SlowFast model for video recognition.

![img1](https://miro.medium.com/max/1400/1*gQIYIEV81TtUjuH-xdH4jg.png)

1. ***Slow* pathway is designed to capture semantic information** that can be given by images or a few sparse frames, and it operates at low frame rates and slow refreshing speed.
2. ***Fast* pathway is responsible for capturing rapidly changing motion**, by operating at fast refreshing speed and high tem- poral resolution. Despite its high temporal rate, this pathway is made very **lightweight**, e.g., ∼20% of total computation. This is because this pathway is designed to have fewer channels and weaker ability to process spatial information, while such information can be provided by the first pathway in a less redundant manner.
   
The two pathways are fused by **lateral connections**.
<br/>

### **Biological derive**
Our method is partially inspired by *biological studies on the retinal ganglion cells* in the primate visual system.   

These studies found that in these cells, ∼80% are Parvocellular (P-cells) and ∼15-20% are Magnocellular (M-cells). **The M-cells operate at high temporal frequency and are responsive to fast temporal changes**, but not sensitive to spatial detail or color.   
**P-cells provide fine spatial detail and color, but lower temporal resolution, responding slowly to stimuli**.
<br/>
<br/>

## Slowfast Networks
---
### **Slow pathway**
The key concept in our Slow pathway is a large temporal stride τ on input frames, i.e., it processes only one out of τ frames. A typical value of τ we studied is 16—this refreshing speed is roughly 2 frames sampled per second for 30-fps videos.   

### **Fast pathway**
The Fast pathway is an- other convolutional model with    
1. High frame rate
   >The Fast pathway works with a small temporal stride of τ/α, where α > 1 is the frame rate ratio between the Fast and Slow pathways., so the Fast pathway samples αT frames, α times denser than the Slow pathway. A typical value is α = 8 in our experiments.

2. High temporal resolution features
   >The feature tensors always have αT frames along the temporal dimension, maintaining temporal fidelity as much as possible.
3. Low channel capacity
   >* The Fast pathway is a convolutional network analogous to the Slow pathway, but has a ratio of β (β < 1) channels of the Slow pathway. The typical value is β = 1/8 in our experiments.   
   In our instantiations, the Fast pathway typically takes ∼20% of the total computation.   
   >* The low channel capacity can also be interpreted as **a weaker ability of representing spatial semantics**.   
   (Reducing input spatial resolution and removing color information.)   

<br/>

### **Lateral connections**
 We attach one lateral connection be- tween the two pathways for every “stage".   
 The two pathways have different temporal dimensions, so the lateral connections perform a transformation to match them.   
 We use unidirectional connections that **fuse features of the Fast pathway into the Slow one**.   
 A global average pooling is performed on each pathway’s output. Then two pooled feature vectors are **concatenated** as the input to the fully-connected classifier layer.
<br/>

### **Instantiations**
### Hyperparameters:   
>S = height and width of a square spatial crop   
T = Temporal length = 4  
α = speed ratio = 8   
β = channel ratio = 1/8    
τ = temporal stride = 16

<br/>

### Transformations in the lateral connections:   
>(i) Time-to-channel: We reshape and transpose {αT, S^2, βC} into {T , S^2, αβC}, meaning that we pack all α frames into the channels of one frame.   
>(ii) Time-strided sampling: We simply sample one out of every α frames, so {αT , S^2, βC} becomes {T , S^2, βC}.  
>(iii) Time-strided convolution: We perform a 3D convolution of a 5×1^2 kernel with 2βC output channels and stride = α.    
>The output of the lateral connections is fused into the Slow pathway by summation or concatenation.

<br/>

## Experiments 1: Action Classification
---
We evaluate our approach on three video recognition datasets(Kinetics-400, the recent Kinetics- 600, and Charades) for action classification.   

<br/>

<img src="https://user-images.githubusercontent.com/31475037/61861644-5065ee00-af07-11e9-9d19-3371565c4ff6.PNG" width="450" height="550"/>
<br/>

![img3](https://blog.airlab.re.kr/assets/images/posts/2019-11-08-SlowFast/figure4.png)

<br/>

### Ablation Experiments:
![img4](https://junsk1016.github.io/img/SlowFast-Table5.JPG)
**5a**   
SlowFast models with various lateral connections:    
time-to-channel (TtoC), time-strided sampling (T-sample), and time-strided convolution (T-conv). For TtoC, which can match channel dimensions, we also report fus- ing by element-wise summation (TtoC, sum).     
For all other variants concatenation is employed for fusion.   

**5c**   
We consider:    
(i) A half spatial resolution (112×112), with β=1/4 (vs. default 1/8) to roughly maintain the FLOPs   
(ii) Gray-scale input frames    
(iii) “Time difference" frames, computed by subtracting the current frame with the previous frame    
(iv) Using optical flow as the input to the Fast pathway
<br/>
<br/>

##  Experiments 2: AVA Action Detection
---
We evaluate our approach on video recognition dataset(AVA dataset) for action detection.   
The performance metric is mean Average Precision (mAP) over 60 classes, using a frame-level IoU threshold of 0.5.
<br/>

![img5](https://junsk1016.github.io/img/SlowFast-Table7.jpg)
<br/>

![img6](https://junsk1016.github.io/img/SlowFast-Fig3.JPG)
The SlowFast model is worse in only 3 categories: “answer phone" (-0.1 AP), “lie/sleep" (-0.2 AP), “shoot" (-0.4 AP), and their decrease is relatively small vs. others’ increase.
<br/>
<br/>

## Conclusion
---
This paper has investigated an architecture design that contrasts the speed along this axis. It achieves state-of-the-art accuracy for video ac- tion classification and detection. We hope that this SlowFast concept will foster further research in video recognition.

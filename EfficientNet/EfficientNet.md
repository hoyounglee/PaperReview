# EfficientNet: Rethinking Model Scaling for Conventional Neural Networks  [[Papaer]](https://arxiv.org/pdf/1905.11946.pdf)
---
<br/> 

## 1. Abstract
---

 In this paper, we systematically study model scaling and identify that
carefully balancing network depth, width, and resolution can lead to better performance.   
Based
on this observation, we propose a new scaling
method that uniformly scales all dimensions of
depth/width/resolution using a simple yet highly
effective compound coefficient.   
To go even further, we use neural architecture search to design a new baseline network
and scale it up to obtain a family of models,
called EfficientNets.  

![fig1](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbKc5oi%2Fbtq17a1fkUK%2Fq9KZqAIax48LJRVtLDHrBK%2Fimg.jpg)
<br/> 
<br/> 

## 2. Introduction
---

In previous work, it is common to scale
only one of the three dimensions – depth, width, and image
size.   
Our empirical study shows that it is critical to balance all
dimensions of network width/depth/resolution, and surprisingly such balance can be achieved by simply scaling each
of them with constant ratio. Based on this observation, we
propose a simple yet effective **compound scaling method**.   
Unlike conventional practice that arbitrary scales these factors, our method uniformly scales network width, depth, and resolution with a set of fixed scaling coefficients.   
 
![fig 2](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbPVTyn%2Fbtq04LoQMCz%2FrfAPx1M0tkux0bFdwfpxbK%2Fimg.png)
<br/> 
<br/> 

## 3. Compound Model Scaling
---

### 3.1 Problem Formulation
Define a ConvNet as:   

![formula](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FC0Uvp%2Fbtq05pTpAxB%2FKEWakjYYrruswz3eE5BuN1%2Fimg.png)

Unlike regular ConvNet designs that mostly focus on finding the best layer architecture Fi
, model scaling tries to expand the network length (Li), width (Ci), and/or resolution
(Hi, Wi) without changing Fi predefined in the baseline network.  
 In order to further reduce the design space, we restrict that all layers must be scaled uniformly with constant ratio. Our target is to maximize the model accuracy
for any given resource constraints, which can be formulated as an optimization problem:   

![formula](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FrXUPg%2Fbtq09yawS1h%2F7lifBkPwKy3CGZrU9EWXGk%2Fimg.png)
<br/> 
<br/> 

### 3.2 Scaling Dimensions
The main difficulty of problem 2 is that the optimal d, w, r
depend on each other and the values change under different
resource constraints. Due to this difficulty, conventional
methods mostly scale ConvNets in one of these dimensions: Depth(d), Width(w), Resolution(r).

![fig 3](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FGTdEy%2Fbtq038q7y94%2FGVjfukxqj5c76wrP59EaC0%2Fimg.png)

>**Observation 1** – Scaling up any dimension of network width, depth, or resolution improves accuracy, but the accuracy gain diminishes for bigger models.   

<br/> 
<br/> 

### 3.3 Compound Scaling
We empirically observe that different scaling dimensions are not independent.  
These intuitions suggest that we need to coordinate and balance different scaling dimensions rather than conventional single-dimension scaling.   

>**Observation 2** – In order to pursue better accuracy and
efficiency, it is critical to balance all dimensions of network
width, depth, and resolution during ConvNet scaling.   

<br/> 

In this paper, we propose a new compound scaling method,
which use a compound coefficient φ to uniformly scales
network width, depth, and resolution in a principled way:   

![formula](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FA3VwW%2Fbtq04LicwqW%2FPWqolLbJOlhJkl3hgvnl1k%2Fimg.png)

where α, β, γ are constants that can be determined by a small grid search. Intuitively, φ is a user-specified coefficient that controls how many more resources are available for model scaling, while α, β, γ specify how to assign these extra resources to network width, depth, and resolution respectively. 

scaling a ConvNet with equation 3 will approximately increase total FLOPS by
(α · β^2 · γ^2)^φ. In this paper, we
constraint (α · β^2 · γ^2) ≈ 2 such that for any new φ, the total FLOPS will approximately3
increase by 2^φ.
<br/> 
<br/> 

## 4. EfficientNet Architecture
---

Since model scaling does not change layer in baseline network, having a good baseline network is also critical.   
We develop our baseline network by leveraging a multi-objective neural architecture search that optimizes both accuracy and FLOPS.
Our search produces an efficient network, which we name EfficientNet-B0.

![efficientnet architecture](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqQzQA%2Fbtq04LihLfG%2FUE8UKgHq7qjqbsjR6OjWkk%2Fimg.png)

 Its main building block is mobile inverted bottleneck MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation optimization.


Starting from the baseline EfficientNet-B0, we apply our
compound scaling method to scale it up with two steps:

>• STEP 1: we first fix φ = 1, assuming twice more resources available, and do a small grid search of α, β, γ
based on Equation 2 and 3. In particular, we find
the best values for EfficientNet-B0 are α = 1.2, β =
1.1, γ = 1.15, under constraint of (α · β^2 · γ^2) ≈ 2.   
>
>• STEP 2: we then fix α, β, γ as constants and scale up
baseline network with different φ using Equation 3, to
obtain EfficientNet-B1 to B7 (Details in Table 2).  

<br/> 
<br/> 

## 5. Experiments
---

In order to further understand why our compound scaling
method is better than others, Figure 7 compares the class
activation map (Zhou et al., 2016) for a few representative
models with different scaling methods.   

![mobilenet, resnet](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FYl6LP%2Fbtq04MIjEfi%2FWumWEdMezK3VL25IkHfIQ0%2Fimg.png)

![fig 5](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FqW9Ss%2Fbtq04MO2C6w%2FzJSi53JeLQJAfz8anNYerk%2Fimg.png)


![performance](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb87rjM%2Fbtq05of3xEZ%2FPC0jl33QYBXm7n7RAz16oK%2Fimg.png)

<br/> 
<br/> 

## 6. Discussion
---

![fig 7](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbbzAgY%2Fbtq2cnGdl5S%2FsKsE9bHxtEZyEokKE04QX1%2Fimg.jpg)

As shown in the figure, the model with compound scaling tends to focus on more relevant regions with more object details, while other models are either lack of object details or unable to capture all objects in the images.

<br/> 
<br/> 

## 7. Conclusion
---

We propose a simple and highly effective compound scaling method, which enables us to easily scale up a baseline ConvNet to any target resource constraints in a more principled way, while maintaining model efficiency.

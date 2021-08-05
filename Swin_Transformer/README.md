
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 0. Abstract
- This new vision transformer capably serves as a general-purpose backbone for computer vision
- The auhtor proposes hierarchical transformer whose representation is computed with shifted windows for solving main challenges of previous transformer which are **large variations in the scale of visual entities** and **high resolution of pixels in images**.
- Shifted windowing scheme brings greater efficiency by **limiting self-attention computation to non-overlapping local windows**.

## 1. Introduction
<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/32179857/124716739-5fe97080-df3f-11eb-9783-6df7c0eecdaa.png">
</p>

- Modeling for computer visions and for NLP has taken a different path in the past. Along with the tremendous success of transformer in NLP domain, researchers started to investigate its adaptation to computer vision.
- The authors argue that significant challenges in transferring its high performance to the visual domain can be explained by difference of two modalities.

    1. Unlike word tokens that serve as the basic elements of processng in NLP tasks, visual elements vary substantially in scale.
    2. various vision tasks such as semantic segmentation require dense prediction at the pixel level, which can computationally expensive quadratically to image size.

- To overcome these issues, the authors propose a general-purpose Transformer backbone, called Swin Transformer.
<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/32179857/124723889-721add00-df46-11eb-9c75-3bd32d1c2075.png">
</p>

#### Characteristics


- Hierarchical representation by starting from small-sized patches and gradually merging neighboring patches in deeper layers.
   --> this can leverage advanced techniques for dense prediction such as FPN or U-Net.
- Key design for Swin Transformer is its **shift of window partition** between consecutive self-attention layers.



## 2. Related work
- **Self-attention based backbone architectures**
    - In these works, the self-attention is computed within a local window of each pixel to expedite optimization, and they achieve slightly better acc.
    - However, costly memory access causes significantly larger latency.
    - ![image](https://user-images.githubusercontent.com/32179857/125021485-b4aef780-e0b5-11eb-8f4e-f1078fda2128.png)
    (https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms)
    
- **Self-attention/Transformers to complement CNNs**
    - Self-attention layers can complement backbones or head networks by providing the capability to encode distant dependencies or heterogeneous interactions.
    - Recently, encoder-decoder design in Transformer has been applied for OD nd Seg. tasks.
    - [Self-attention Transformer Encoder](https://theaisummer.com/transformer/#self-attention-the-transformer-encoder)
- **Transformer based vision backbones**
    - Vision Transformer (ViT) and its follow-ups.
    - ViT directly applies a Transformer architecture on non-overlapping medium-sized image patches for classification.
    - ViT performs well but requries large dataset.
    - DeiT improves ViT so that it can use ImageNet-1K too. 
    - Left drawback here is lack of 'general purpose' network.
- **Rise of the best speed-accuracy tradeoff architecture, Swin Transformer.**

## 3. Method
#### 3.1 Overall Architecture
<p align="center">
  <img width="1200" height="400" src="https://user-images.githubusercontent.com/32179857/124725841-2b2de700-df48-11eb-9cca-325a8d775488.png">
</p>

1. It first splits an input RGB image into non-overlapping patches b a patch splitting module, like ViT.
    (Each patch is treated as 'Token' and its feature is set as a concat. of the raw pixel RGB values.)
2. Several Transformer blocks with modified self-attention (Swin Transformer blocks) are applied on these patch tokens.
    (This block maintain the number of tokens as (H/4 x W/4), and referred as **'Stage1'**.
3. To produce hierarchical representation, the number of token is reduced by patch merging layers as the network gets deeper.
    (This layer concats neighboring 2x2 patches into one patch, making the number of channels double which is 'Down sampling'. As the network goes deeper,
    since the feature map size which pass each stage gets smaller, and it can play a role as 'hierarchical representation' such as FPN, UNet)
4. It continues each stage with a same process, with different amount of iterations ([2,2,6,2])
- Patch partitioning and Patch merging, Linear embedding
    - patch partition & patch merging

        ![image](https://user-images.githubusercontent.com/32179857/125008196-fa12fb00-e09c-11eb-90b7-76adf2dd83ca.png)
    - Linear embedding

        ![image](https://user-images.githubusercontent.com/32179857/125008344-4c541c00-e09d-11eb-8811-dd194d5c8dc0.png)


- Swin Transformer block
    - Swin Transformer block is built by replacing the standard multi-head self attention (MSA) module by a module based on shifted windows, with other layers kept same.
    - It consists of a shifted window based MSA module, followed by a 2-layer MLP(multi layer perceptron) with GELU non-linearity in between.
    - [Multi-head Attention](https://theaisummer.com/transformer/#the-core-building-block-multi-head-attention-and-parallel-implementation)
    
        ![image](https://user-images.githubusercontent.com/32179857/125007362-fbdbbf00-e09a-11eb-8cd0-4d7cb65f9be4.png)
        ![image](https://user-images.githubusercontent.com/32179857/125007379-06965400-e09b-11eb-8c47-23c051108261.png)

#### 3.2 Shifted Window based Self-Attention
- The standard Transformer and its adaptation for image classification both conduct global self-attention, where the relationships between a token and all other tokens are computed. --> this leads to quadratic complexity with respect to the number of tokens (not suitable for vision tasks due to high resolution)

##### Self attention with non-overlapping windows
- The authors propose efficient self-attention within local windows
- computational complexity
    - Suppose each window contains M x M patches, 
        - global MSA: 

        ![image](https://user-images.githubusercontent.com/32179857/125007742-e31fd900-e09b-11eb-98c4-7a37f2234784.png)
    - the former(standard Transformer) is quadratic to patch number hw
    - the latter in linear when M is fixed.
    - since the patch size M is much smalled than image resolution h,w, the computational cost is also much cheaper.
##### Shifted window partitioning in successive blocks
- The window-based self-attention module lacks connections across the windows
- To solve this, the authors propose a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks
    - 1st module uses reguar window partitioning
    - 2nd module adopts a windowing configuration that is shifted from that of the preceding layer by displacing the windows by (M/2, M/2) pixels

        ![image](https://user-images.githubusercontent.com/32179857/125009292-39424b80-e09f-11eb-846c-b594347f951a.png)

#### 3.3 Architect Variants
```
• Swin-T: C = 96, layer numbers = {2, 2, 6, 2}
• Swin-S: C = 96, layer numbers ={2, 2, 18, 2}
• Swin-B: C = 128, layer numbers ={2, 2, 18, 2}
• Swin-L: C = 192, layer numbers ={2, 2, 18, 2}
```

## Experiments
- experiments conducted with ImageNet-1K image classification / COCO object detection, ADE20K semantic segmentation.
<p align="center">
  <img width="800" height="600" src="https://user-images.githubusercontent.com/32179857/124850442-94ab0580-dfdb-11eb-950c-2a694030c4da.png">
</p>

1. Classification
   - AdamW optimizer
   - 300 epoch
   - 20 epoch learning rate warmup
   - batch 1024
   - ImageNet-22K pretrained


2. Object Detection
    - Multi-scale training
    - AdamW optimizer
    - batch 16
   
   
### Ablation Study
<p align="center">
  <img width="800" height="500" src="https://user-images.githubusercontent.com/32179857/124851005-a8a33700-dfdc-11eb-9593-195bf6a19c75.png">
</p>

&nbsp;


### References
- https://byeongjo-kim.tistory.com/36
- https://www.youtube.com/watch?v=AA621UofTUA
- https://mlfromscratch.com/activation-functions-explained/#/
- https://wikidocs.net/31379
- https://visionhong.tistory.com/31
- https://github.com/microsoft/Swin-Transformer
- https://xzcodes.github.io/posts/paper-review-swin-transformer


# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

## 0. Abstract
- This new vision transformer capably serves as a general-purpose backbone for computer vision
- The auhtor proposes hierarchical transformer whose representation is computed with shifted windows for solving main challenges of previous transformer which are **large variations in the scale of visual entities** and **hight resolution of pixels in images**.
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
- **Self-attention/Transformers to complement CNNs**
    - Self-attention layers can complement backbones or head networks by providing the capability to encode distant dependencies or heterogeneous interactions.
    - Recently, encoder-decoder design in Transformer has been applied for OD nd Seg. tasks.
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

[ TBD ]

#### Shifted Window based Self-Attention

#### Architect Variants
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


3. Sementic Segmentation
<p align="center">
  <img width="400" height="400" src="https://user-images.githubusercontent.com/32179857/124850628-f4a1ac00-dfdb-11eb-97be-355f1271b5b4.png">
</p>
   
   
### Ablation Study
<p align="center">
  <img width="800" height="500" src="https://user-images.githubusercontent.com/32179857/124851005-a8a33700-dfdc-11eb-9593-195bf6a19c75.png">
</p>


## Conclusion
[TBD]

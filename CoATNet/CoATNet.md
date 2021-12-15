# CoAtNet: Marrying Convolution and Attention for All Data Sizes

## 1. Introduction
Since the breakthrough of AlexNet, Convolutional Neural Networks (ConvNets) have been the dominating model architecture for computer vision.    
Meanwhile, with the success of self-attention models like Transformers in natural language processing, many previous works have attempted to bring in the power of attention into computer vision.
While ViT has shown impressive results with enormous JFT 300M training images, its performance still falls behind ConvNets in the low data regime.
ViT achieves comparable results to state-of-the-art (SOTA) ConvNets, indicating that Transformer models potentially have higher capacity at scale than ConvNets.   
In this work, we systematically study the problem of hybridizing convolution and attention from two fundamental aspects in machine learning – generalization and model capacity.    
**a key challenge here is how to effectively combine them to achieve better
trade-offs between accuracy and efficiency**


First, we observe that the commonly used depthwise convolution can be effectively merged into attention layers with simple relative attention.   
Second, simply stacking convolutional and attention layers, in a proper way, could be surprisingly effective to achieve better generalization and capacity.

</br>

## 2. Model
**how to “optimally” combine the convolution and transformer**

>1. How to combine the convolution and self-attention within one basic computational block?
>2. How to vertically stack different types of computational blocks together to form a complete network?

### 2.1 Merging Convolution and Self-Attention

For convolution, we mainly focus on the MBConv block which employs depthwise convolution to capture the spatial interaction. A key reason of this choice is that both the FFN module
in Transformer and MBConv employ the design of “inverted bottleneck”, which first expands the channel size of the input by 4x and later project the the 4x-wide hidden state back to the original channel size to enable residual connection. 

<img src="https://user-images.githubusercontent.com/48341349/141216119-4476844b-17a0-46e5-a2b5-3a77781a00cf.png" width="500" height="100"/>


where xi , yi ∈ R^D are the input and output at position i respectively, and L(i) denotes a local neighborhood of i, e.g., a 3x3 grid centered at i in image processing.

In comparison, self-attention allows the receptive field to be the entire spatial locations and computes
the weights based on the re-normalized pairwise similarity between the pair (xi, xj):

<img src="https://user-images.githubusercontent.com/48341349/141216121-307abd5f-6f99-41c5-b458-3ab8a54b20a3.png" width="500" height="140"/>


A straightforward idea that could achieve this is simply to sum a global static convolution kernel with the adaptive attention matrix, either after or before the Softmax normalization, i.e.

<img src="https://user-images.githubusercontent.com/48341349/141216116-d533e22d-1ef6-4556-b9fe-1bb22af9337c.png" width="750" height="100"/>

<img src="https://user-images.githubusercontent.com/48341349/141216106-ea00905b-da6e-49c1-b49a-a1d2572239fc.png" width="500" height="140"/>

Given the comparison above, an ideal model should be able to combine the 3 desirable properties in Table 1. With the similar form of depthwise convolution in Eqn. (1) and self-attention in Eqn. (2), a
straightforward idea that could achieve this is simply to sum a global static convolution kernel with the adaptive attention matrix, either after or before the Softmax normalization, i.e

</br>

### 2.2 Vertical Layout Design

After figuring out a neat way to combine convolution and attention, we next consider how to utilize it to stack an entire network.

The global context has a quadratic complexity w.r.t. the spatial size.  
Hence, to construct a network that is feasible in practice, we have mainly three options:   
(A) Perform some down-sampling to reduce the spatial size and employ the global relative attention
after the feature map reaches manageable level.   
(B) Enforce local attention, just like in convolution.   
(C) Replace the quadratic Softmax attention with certain linear attention variant 

For option (A), the down-sampling can be achieved by either (1) a convolution stem with aggressive
stride (e.g., stride 16x16) as in ViT or (2) a multi-stage network with gradual pooling as in ConvNets.
With these choices, we derive a search space of 5 variants and compare them in controlled experiments.

• When the ViT Stem is used, we directly stack L Transformer blocks with relative attention, which
we denote as VIT(REL).   
• When the multi-stage layout is used, we mimic ConvNets to construct a network of 5 stages (S0,
S1, S2, S3 & S4), with spatial resolution gradually decreased from S0 to S4. At the beginning
of each stage, we always reduce the spatial size by 2x and increase the number of channels (see
Appendix A.1 for the detailed down-sampling implementation).

>S0: a simple 2-layer convolutional Stem   
S1: MBConv blocks with squeeze-excitation (SE)   
S2~S4: either the MBConv or the Transformer block
>
>VIT(REL), C-C-C-C, C-C-C-T, C-C-T-T and C-T-T-T

</br>

* Generalization capability
 
<img src="https://creamnuts.github.io/assets/images/short_review/coatnet/image-20210824172433335.png" width="550" height="300"/>

>C-C-C-C ≈ C-C-C-T ≥ C-C-T-T > C-T-T-T ≫ ViT(REL)

Particularly, VIT(REL) is significantly worse than variants by a large margin, which we conjecture
is related to the lack of proper low-level information processing in its aggressive down-sampling Stem. 

</br>

* Model Capability

<img src="https://creamnuts.github.io/assets/images/short_review/coatnet/image-20210824172559116.png" width="550" height="300"/>

>C-C-T-T ≈ C-T-T-T > ViT(REL) > C-C-C-T > C-C-C-C

Importantly, this suggests that simply having more Transformer blocks does NOT necessarily mean higher capacity for visual processing.    
On one hand, while initially worse, VIT(REL) ultimately catch up with the two variants with more MBConv stages, indicating the capacity advantage of Transformer blocks. On the other hand, both C-C-T-T and C-T-T-T clearly outperforming VIT(REL) suggest that the ViT stem with an aggressive stride may have lost too much information and hence
limit the model capacity.

</br>

Finally, to decide between C-C-T-T and C-T-T-T, we conduct another transferability test.

<img src="https://creamnuts.github.io/assets/images/short_review/coatnet/image-20210824173716401.png" width="550" height="140"/>


Taking generalization, model capacity, transferability and efficiency into consideration, we adapt the C-C-T-T multi-stage layout for CoAtNet. 

![](https://creamnuts.github.io/assets/images/short_review/coatnet/image-20210824174452402.png)

</br>

## 3. Experiments

<img src="https://user-images.githubusercontent.com/48341349/141215691-9177cd99-e098-45fb-bc19-134c9814f816.jpg" width="700" height="400"/>

<img src="https://user-images.githubusercontent.com/48341349/141215706-70cbdc4c-e4b8-4e52-8db1-326c0eecffe2.png" width="650" height="220"/>

<img src="https://user-images.githubusercontent.com/48341349/141215698-2cd5b57e-4bec-4e4a-af6a-f386c7f50ed4.png" width="650" height="800"/>


</br>

## 4. Conclusion
Extensive experiments show that CoAtNet enjoys both good generalization like ConvNets and superior model capacity like Transformers, achieving state-of-the-art performances under different data sizes and computation budgets.

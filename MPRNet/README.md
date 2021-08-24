# MPRNet (https://arxiv.org/abs/2102.02808, CVPR2021)

## Abstract
- The authors propose a multi-stage architecture that progressively learns restoration functions for the degraded inputs
    - The model first learns the **contextualized feature** using encoder-decoder architectures
    - Later combines them with a high-resolution branch that retains **local information.**
- To this end, the authors propose a two faceted approach where the information is not only **exchanged sequentially** from early to late stages, but **lateral connections between feature processing blocks** also exist to avoid any loss of information


## Introduction
<p align="center">
  <img width="350" height="280" src="https://user-images.githubusercontent.com/32179857/130537416-81ea360d-3a46-4393-ac52-d9d721d42255.png">
</p>

The authors analyized these days' efforts that have been made to bring the multi-stage design to image deblurring, deraining, etc.
1. Existing encoder-decoder architecture in multi-stage techniques is effective in encoding broad contextual information, but **unreliable in preserving spatial image details**, and single-scale pipeline is spatially accurate but **semantically less reliable**.
    -> they showed that the combination of two in multi-stage is needed.
2. Naively passing the output of one stage to the next stage yields **suboptimal results.**
3. **Providing ground-truth supervision** at each stage for progressive restoration is important.
4. A mechanism to propagate intermediate features from earlier to later stages is required to preserve contextualized features.

Thus, the authors introduce the main contributions below.
1. A novel multi-stage approach capable of generating contextually-enriched and spatially accurate outputs.
2. An effective supervised attention module that takes full advantage of the restored image at every stage.
3. A strategy to aggregate multi-stage features across stages.


## Related Works
1. Single-stage Approaches
    the majority of image restoration methods are based on a single-stage design and the architecture components are usually based on those developed for high-level vision tasks.
    - residual learning
    - encoder-decoder 
    - dilation convolution
2. Multi-stage Approaches
    - Such a design is effective since it decomposes the challenging image restoration task into smaller easier subtasks.
    - However, thismay yield suboptimal results
3. Attention
    - Driven by its success in high-level tasks, it has been used in low-level vision tasks.


## Multi-Stage Progressive Restoration
<p align="center">
  <img width="500" height="500" src="https://user-images.githubusercontent.com/32179857/130538012-4b8870ca-591d-41c6-a676-38322f30b029.png">
</p>

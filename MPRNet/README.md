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

- The proposed framework for image restoration, shown in Fig.2, consists of three stages.
    - The first two stages are based on **encoder-decoder** subnetworks that **learn the broad contextual information** due to large receptive fields
    - Since image restoration is position-sensitive, the last stage employs a subnetwork that operates on the original input image resolution
      (This maintains pixel-to-pixel correspondence from the input to output)
- Incorporated a **supervised attention module** between every two stages.
    - With the supervision of GT images, the module rescales the feature maps of the previous stage before passing them to the next stage.
- Although MPRNet stacks multiple stages, each stage has an access to the input image.

~~~
- The authors adapt multi-patch hierarchy on the input image and split the image into non-overlapping patches
    - four for stage1, two for stage2, and original image for last stage.
- Instead of directly predicting a restored image X_s, the model predicts a residual image R_s --> X_s = I + R_s
~~~

<p align="center">
  <img width="500" height="400" src="https://user-images.githubusercontent.com/32179857/130543593-465bb10f-ee97-403c-8fa7-e4927104360d.png">
</p>

#### Complementary Feature Processing 
- Existing single-stage CNN for image restoration typicall use -> 1)encoder-decoder 2) single-scale feature pipeline
- encoder-decoder
    - they are prone to sacrificing spatial details due to the repeated use of downsampling operation.
- single-scale feature pipeline
    - they are semantically less robust due to the limited receptive field.

##### Encoder-Decoder subnetwork
- herein, the authors propose subnetwork which is based on standard U-Net
    1) added channel attention blocks(CAB) to extract features at each scale.
    <p align="center">
      <img width="600" height="200" src="https://user-images.githubusercontent.com/32179857/130709251-6d408570-3e52-45ca-ab10-b9dfec90e3f9.png">
    </p>

    2) skip connection in U-Net is also processed with the CAB.
    3) Transposed conv --> bilinear upsampling followed by a conv layer.
    
##### Original Resolution subnetwork
- to preserve fine details from the input image, the authors introduce ORSNet which does not employ any downsampling operation and generates spatially-enriched high-resolution features.
<p align="center">
  <img width="1000" height="300" src="https://user-images.githubusercontent.com/32179857/130880201-ef7dd976-6d65-45e2-b38e-41261063d994.png">
</p>


#### Cross-stage Feature Fusion
- CSFF Module between two encoder-decoders.
- Features from one stage are first refined with 1x1 conv for aggregation
##### merits
    1. It makes the network less vulnerable by the information loss due to repeated use of up-down sampling operations in encoder-decoder.
    2. The multi-scale fetures of one stage help enriching the features of the next stage.
    3. The network optimization procedure becomes more stable as it eases the flow of information.

#### Supervised Attention Module
- It provides G.T. supervisory signals useful for the progressive image restoration at each stage.
- it suppress the less informative features at the current stage and only allow the useful ones to propagate to the next stage.

<p align="center">
  <img width="600" height="300" src="https://user-images.githubusercontent.com/32179857/130881393-f1a2e385-ffec-4746-b3ec-73fba8e9df51.png">
</p>

1. SAM takes the incoming feature of the eariler stage and first generates a residual image with 1x1 conv.
2. The residual image is added to the degraded input image I to obtain the restored image X_s
3. To predict X_s, the authors provide explicit supervision with the G.T. image
4. per-pixel attention masks are generated from the image X_s using 1x1 conv followed by the sigmoid.
5. the attnetion augmented feature representation F_out, produced by SAM, is passed to the next stage for further processing. 

## Experiments
1. Deraining
![image](https://user-images.githubusercontent.com/32179857/130883469-a71773d2-7c27-4a79-a2a8-550182169e09.png)
* PSNR: Peak Signal-to-Noise ratio
* SSIM: Structual Simiarity Index

![image](https://user-images.githubusercontent.com/32179857/130883776-b218bd74-50ec-4b8a-ad2b-24c5fba354fc.png)

2. Deblurring
<p align="center">
  <img width="500" height="500" src="https://user-images.githubusercontent.com/32179857/130883847-55a47e53-74a2-49fc-be44-9cf287081053.png">
</p>

3. Deblurring on RealBlur Dataset
<p align="center">
  <img width="500" height="500" src="https://user-images.githubusercontent.com/32179857/130883912-6f35ad3b-fe4b-4ba9-b477-9245ea3cb9eb.png">
</p>
![image](https://user-images.githubusercontent.com/32179857/130884024-626ff5a5-e7dc-4bc5-9592-adc1d4aa621f.png)


4. Denoising
![image](https://user-images.githubusercontent.com/32179857/130884089-fb3c4377-4857-4122-98f6-bfcd6683a07a.png)



### References
- https://github.com/swz30/MPRNet
- https://arxiv.org/pdf/2102.02808v2.pdf





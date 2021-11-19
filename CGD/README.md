# CGD (Combination of Multiple Global Descriptors for Image Retrieval)


## Abstract
- Current image retrieval researches show that ensemble multiple models with various global descriptors improves the model performance.
- However, training multi models for ensemble takes too much time.
- This paper propose new framework, which is end-to-end trainable global descriptors for ensemble.
- This framework is flexible and easilly extendable.


## Introduction
- General Image Retrieval researches had used FC layer after conv layer which decrease image dimension for global descriptor.
- Bunch of researches about global descriptors
    - Global pooling method (using convolution layer's activation)
        - SPoC (Sum Pooling of Convolution)
        - MAC (Maximum Activation of Convolution)
        - GeM (Generalized mean Pooling)

        - These Global Descriptors have different characteristics so that it change the performance depending on the dataset
            (ex) SPoC works well at large region in an image, MAC works well at local focused region in an image.
            
    - Some chages on Global Pooling method
        - weighted sum pooling
        - weighted GeM
        - R-MAC (residual MAC)
        - etc

- This paper focues on the method that can work like ensembled model which does not have to consider dataset variations.
- The paper proposes end-to-end combination of global descriptors that can work as ensembled.
- Very flexable, and expandable depending on global descriptor, CNN backbone, loss, dataset.
    - ex: baseline + attention module + focal loss + etc
- With the method, it achived SOTA in numerous Image Retrieval tasks (CARS 196, CUB200, SOP, In-Shop)


## Proposed Framework
- CGD has a framework which is concatenation of multiple global descriptors.
- The method has CNN backbone and two modules.
    - 1st module is main module, which learns image representation using ranking loss.
    - 2nd mudule is sub-module, which supports to finu-tune CNN with classification loss.
 - final loss: ranking loss + classification loss = final loss.
 - model architecture
 
   ![image](https://user-images.githubusercontent.com/32179857/142548224-61c86e8b-4863-450c-839e-6835a3525de0.png)


### Backbone
- Any backbone can be applied! --> here the authors use ResNet-50
- To keep feature map information of last layer, the method removed downsampling part between stage(block)3 and stage(block)4.
- Input size = (224x224), output feature map size = (14x14)

### Main Module: Multiple Global Descriptors


### References
https://simonezz.tistory.com/96
https://arxiv.org/pdf/1903.10663.pdf
https://cyc1am3n.github.io/2019/05/02/combination-of-multiple-global-descriptors-for-image-retrieval.html




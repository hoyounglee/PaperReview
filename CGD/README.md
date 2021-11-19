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


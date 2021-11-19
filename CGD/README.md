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
- The output of the main module is the image representation of global decriptors which is attached to final conv. layer.
- In the paper, the authors use SPoC, MAC, GeM.
- Given an image I, the output of the final layer C x H x W 3D tensor(we call it as **'x'**)
- χc is the activation result set here (feature map c ∈ {1...C}).
- global descriptor gets **x** as input and generate vector **f** by taking pooling process.
  ![image](https://user-images.githubusercontent.com/32179857/142573169-293377fb-230b-4816-b2ab-f859380cd487.png)
- The global descriptor is decided by the term ![image](https://user-images.githubusercontent.com/32179857/142573292-a3a9c2db-667c-452a-b0cb-62b0f332795b.png).
    - ![image](https://user-images.githubusercontent.com/32179857/142573328-fb4176a6-2b45-4cba-b0a7-3863df149439.png) = 1    : SPoC
    - ![image](https://user-images.githubusercontent.com/32179857/142573328-fb4176a6-2b45-4cba-b0a7-3863df149439.png) = ∞    : MAC
    - ![image](https://user-images.githubusercontent.com/32179857/142573328-fb4176a6-2b45-4cba-b0a7-3863df149439.png) = else : GeM
- ![image](https://user-images.githubusercontent.com/32179857/142573328-fb4176a6-2b45-4cba-b0a7-3863df149439.png) fir GeM is either trainable or fixed. (here, 3)
- The feature vector from i'th branch after FC layer and l2-normalization is below.

    ![image](https://user-images.githubusercontent.com/32179857/142574036-13e0a9e5-2433-49ef-9487-ea2b9c6962c9.png)
    
    (n is number of branch, i ∈ {1...n})
    (Wi is weight of FC layer)
- Final feature vector, which is combined descriptor ![image](https://user-images.githubusercontent.com/32179857/142574182-e05b07d6-7b42-463d-8c88-68615b0880ca.png) merges branches (here 3), and perform l2-norm.

    ![image](https://user-images.githubusercontent.com/32179857/142574264-7692fa95-c6f9-438e-8c2a-c5fdc024089e.png)

- loss for this global descriptor is batch-hard triplet loss
    ##### example of triplet loss
    
    ![image](https://user-images.githubusercontent.com/32179857/142575109-cf79a4df-3ad8-4d4b-a481-047329deb371.png)

    ##### Batch hard strategy
    - In this strategy, we want to find the hardest positive and negative for each anchor.
    
    ![image](https://user-images.githubusercontent.com/32179857/142575246-71642f51-a8aa-46b7-bca8-72685ff57b5c.png)

- Thie framework has 2 benefits mainly.
    - ensemble effect by adding few parameters.
    - expandable / widley applicable without diversity control, by using each branch output.

### Auxiliary Module: Classification Loss
- Sub-module is using first global descriptor's output from main module.
- This approach is conducted by two procedures.
    - fine-tune the backbone with classification loss for improving conv. filter's performance.
    - fine-tune the network for improving global descriptor.
- In here, the authors concatenated both in end-to-end.
- By training auxiliary loss, we can,
    - generate image representation which has a characteristic between classes, 
    - train the network fast
    - more stable than training main module's ranking loss.
- The authors also adapt tenperature scaling and label smoothing.
- softmax loss with temperature scaling

    ![image](https://user-images.githubusercontent.com/32179857/142576273-6cbd6444-db99-4c76-9803-efa82ce02bab.png)
    (N is # of classes, y = label, W = trainable weight, b = bias, tou = temperature parameter (default=1), f = global descriptor from first branch)
- This prevents over-fitting too.


## Experiments
![image](https://user-images.githubusercontent.com/32179857/142576667-5f5c4840-ad13-40be-86fc-5d9b0e61e449.png)

![image](https://user-images.githubusercontent.com/32179857/142576638-167f0b07-2bd0-4984-916b-b0e569bf6bb1.png) ![image](https://user-images.githubusercontent.com/32179857/142576707-20631338-ac43-4f8b-aa04-0b61e6b6cf78.png)

#### Conclusion
- Merged loss (Ranking loss + classification loss) > ranking loss
- Trick (label smoothing + temp. scaling) > only temp. saling > only label smoothing > nothing
- CGD > architecture B > architecture A

    ![image](https://user-images.githubusercontent.com/32179857/142576902-ef69c568-e243-4950-8df5-3c27baca2fce.png)
    
    
    




### References
https://simonezz.tistory.com/96
https://arxiv.org/pdf/1903.10663.pdf
https://cyc1am3n.github.io/2019/05/02/combination-of-multiple-global-descriptors-for-image-retrieval.html




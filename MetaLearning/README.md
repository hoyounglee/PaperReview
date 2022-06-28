# Few Shot Learnings
- Despite the high accuracy and speed of recent SOTA algorithms, 
  there is one big issue: for a good-performing solution, we need a huge amount of data. 
- In addition, the data must be annotated, which requires a lot of manual work. That was the reason for the development of several new paradigms like **self-supervised learning** and **few-shot learning.**
- Recent progress in the few-shot classification helped to significantly improve the performance of “learn to learn” problem in classification.
- However, few-shot object detection (FSOD) has large potential to grow and improve.

## Few Shot variations
1. N-Shot Learning (NSL)
2. Few-Shot Learning
3. One-shot Learning (OSL)
4. Zero-Shot Learning (ZSL)

## Few Shot Learning approaches

### N-way-K-Shot-classification
1. A training (support) set that consists of:
   1) N class labels
   2) labeled images for each class (a small amount, less than ten samples per class)
2. Q query images

![image](https://user-images.githubusercontent.com/32179857/165237845-0a742519-1034-45af-a280-693d0795ae60.png)
- this Few-Shot Learning mechanism is characterized as a **Meta-Learning** problem.
- Generally, there are two approaches that you should consider when solving FSL problems:
    - Data-level approach (DLA)
    - Parameter-level approach (PLA)

### Data-level approach
- Data augmentation
- GAN

### Parameter-level approach
- Regularization
- use of proper loss fuctions
- So-called Meta-Learning

## Meta-Learnin algorithms
- Metric Learning
- Gradient-Based Meta-Learning
    - need meta-learner & base-learner
    - meta-learner
       - learns across the episodes
    - base-learner
       - a model that is initialized and trained insdie each episode by the meta-learner

![image](https://user-images.githubusercontent.com/32179857/165244409-f5682c84-7015-4020-9c8c-bee27dc2094d.png)

-Imagine an episode of Meta-training with some classification task defined by a N * K images support-set and a Q query set:
```
1. We choose a meta-learner model,
2. Episode is started,
3. We initialize the base-learner (typically a CNN classifier),
4. We train it on the support-set (the exact algorithm used to train the base-learner is defined by the meta-learner),
5. Base-learner predicts the classes on the query set,
6. Meta-learner parameters are trained on the loss resulting from the classification error,
6. From this point, the pipeline may differ based on your choice of meta-learner.
```
![image](https://user-images.githubusercontent.com/32179857/168216911-b5d1ea7a-ec60-4d26-b965-3e8e9ba24434.png)


## Few-Shot classification methods
1. Model-Agnostic Meta-Learning (MAML)
2. Matching Networks
3. Prototypical Networks
4. Relation Network

### Model-Agnostic Meta-Learning(MAML)
- **MAML** is based on the **Gradient-Based Meta-Learning (GBML)** concept.
- **GBML** is about the meta-learner acquiring prior experience from training the base-model and learning the common features representations of all tasks.
- **MAML** provides a good initialization of a **meta-learner**’s parameters to achieve optimal fast learning on a new task with only a small number of gradient steps while avoiding overfitting that may happen when using a small dataset.

Here is how it’s done:
```
1. The meta-learner creates a copy of itself (C) at the beginning of each episode,
2. C is trained on the episode (just as we have previously discussed, with the help of base-model),
3. C makes predictions on the query set,
4. The loss computed from these predictions is used to update C,
5. This continues until you’ve trained on all episodes.
```
![image](https://user-images.githubusercontent.com/32179857/165249658-9539d7a1-15d6-4ae5-988b-9407a04eed90.png)

### Matching Networks
- Metric Learning based methods
```
1. Each image from the support and the query set is fed to a CNN that outputs embeddings for them,
2. Each query image is classified using the softmax of the cosine distance from its embeddings to the support-set embeddings,
3. The Cross-Entropy Loss on the resulting classification is backpropagated through the CNN.
```

### Prototypical Networks
- Similar to Matching Networks, but uses Euclidean distance instead of cosine distance.

### Relation Network
- Matching Network / Prototypical Networks --> Relation Network
- Overall structures
    - The relation module is put on the top of the embedding module, which is the part that computes embeddings and class prototypes from input images.
    - The relation module is fed with the concatenation of the embedding of a query image with each class prototype, and it outputs a relation score for each couple. Applying a Softmax to the relation scores, we get a prediction.
![image](https://user-images.githubusercontent.com/32179857/165253877-1db8d26e-0173-4ccc-bcea-64f28bdf8486.png)

<img width="787" alt="image" src="https://user-images.githubusercontent.com/32179857/168218788-befd2df4-de3b-4394-b6ad-9192655c7511.png">

## Few Shot Object Detection
#### BenchMark
Two popular few shot object detection tasks are used for benchmark: MS-COCO on 10-shot and MS-COCO on 30-shot. Let’s look at the top 3 models for each of these tasks:
![image](https://user-images.githubusercontent.com/32179857/165255408-177e85d8-8aac-48a1-8861-bccdcb17e3e9.png)

### DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection
- The model modifies Faster R-CNN for few-shot OD.
![image](https://user-images.githubusercontent.com/32179857/165257005-023d703a-769b-4a09-aac7-3debfc21f0b7.png)
- To modify Faster R-CNN to work on few-shot settings authors are trying to solve two problems:

1. **The problem of multi-task learning:** R-CNN head of model is responsible for classification, in other words, what to look at, whilst RPN head aims to understand where to look, it solves localization problem. “First head needs translation invariant features whereas the localization head needs translation covariant features”[1]. The joint optimization of these two heads in the case of FSOD can lead to worse results when we have individual small tasks.
2. **The problem of shared backbone:** As we can see in the image above, Faster R-CNN has one shared backbone for 2 heads. It works very well in object detection, but in a few shot settings, there can be a decrease in accuracy when fine-tuning for novel classes. Foreground-background confusion can arise, which means background in base training can become foreground in the novel fine-tuning phase. That is why gradients from RPN cause overfitting of shared backbone and model can’t converge.

- To solve these problems, the authors suggest changing the model by adding two modules:
   1. **Gradient Decoupled Layers (GDL)**
   2. **Prototypical Calibration Block (PCB)**
![image](https://user-images.githubusercontent.com/32179857/165259610-0c948e63-6fd3-4fd8-98c5-356215f8ca04.png)

### Meta-DETR: Image-Level Few-Shot Object Detection with Inter-Class Correlation Exploitation
- RCNN based models have major problems below
    1. The problem of region proposals: This may work well on a large number of images however under a few-shot setup we have only a limited number of examples per class. Moreover, we try to generalize on novel classes, which makes it harder to get high-quality region proposals.
    2. The problem of poorly defined meta-learning tasks. Each support class is treated independently which causes the problem of well distinguishing similar classes like bikes and motorcycles, cows and sheep, etc (Image 5).
<img width="757" alt="image" src="https://user-images.githubusercontent.com/32179857/168219948-343e558a-e3a4-49a7-b3b2-7c089ad08a72.png">

- To solve the problem with high correlation between similar classes, the authors suggest a new module called **Correlational Aggregation Module (CAM). **
- It aggregates query features with support classes for class-agnostic prediction
- CAM
    - CAM first matches the query features with a set of support classes.
    - Then it maps the set of support classes to a set of pre-defined task encodings that differentiate these support classes in a class agnostic manner.
    - CAM outputs support-aggregated query features which then become the input of a transformer-based model for object detection.
- Hungarian Loss was used for the model, the same as for Deformable DETR. Additionally, cosine similarity cross-entropy is used after CAM to classify class prototypes.

<img width="787" alt="image" src="https://user-images.githubusercontent.com/32179857/168219856-b2aae0e6-8ac2-401f-ae30-906c43c260da.png">

## References
https://neptune.ai/blog/understanding-few-shot-learning-in-computer-vision
https://towardsdatascience.com/review-on-few-shot-object-detection-185965e0e6a2
https://kicarussays.tistory.com/24

[1] Qiao, L., Zhao, Y., Li, Z., Qiu, X., Wu, J., & Zhang, C. DeFRCN: Decoupled Faster R-CNN for Few-Shot Object Detection (2021).| Github
[2]Chen, T., Liu, Y., Su, H., Chang, Y., Lin, Y., Yeh, J., & Hsu, W.H. Dual-Awareness Attention for Few-Shot Object Detection (2021).| Github
[3] Zhang, G., Luo, Z., Cui, K., Lu, S., Meta-DETR: Image-Level Few-Shot Object Detection with Inter-Class Correlation Exploitation (2021).| Github

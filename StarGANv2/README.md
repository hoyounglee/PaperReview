# StarGAN v2: Diverse Image Synthesis for Multiple Domains
- linked model review from [Suzy's StarGANv1 review](https://github.com/geappliances/AI.PaperReview/blob/master/Suzy/StarGAN.md)
- Apr, 2020, from [NAVER CLOVA](https://clova.ai/en/research/research-areas.html)

&nbsp;

## Abstract
- A good I2I translation model should satisfy,
  - diversity of generated images
  - scalability over multiple domains. 

&nbsp;

## Introduction
### < Some pre-indtroduced terms >
  - Domain: a set of images that can be grouped as a visually distinctive category
  - Style: a unique apperance that each image has 
  (ex:  we can set image domains based on the gender of a person, in which case the style includes makeup, beard, and hairstyle)
  
- So far, previous works have approached I2I by injecting a low-dimensional latent code to the generator, which can be randomly sampled from the standard Gaussian distribution. 
- But this can only deal with I2I between two domains.
- If we have K domains, it means that we need to ***train K(K-1) generators*** to handle translations between each and every domain.

### < Author's first trial >
#### StarGAN
  - The generator takes a domain label as an additional input
  - learns to transform an image into the corresponding domain.
  ```diff
  - But it still cannot capture the multi-modal nature of the data distribution due to the fact that,
    - Each domain is indicated by a predetermined label.
    - Since the generator gets fixed label as input, it inevitably produces the same output per each domain.
  ```
  
### < Solution for the issue >
#### StarGANv2
- Starting from StarGAN,
- Replaced domain label with the proposed domain-specific style code that can represent diverse styles of specific domain.
- Introduced two modules, **1. a mapping network, 2. style encoder**
  - Mapping network learns to transform random Gaussian noise into a style code.
  - Encoder learns to extract the style code from a given reference image.
- Both modules have multiple output brances, each of which provides style codes for a specific domain.
- Finally, using these style codes, the generator learns to synthesize diverse images over multi-domains.

&nbsp;

## StarGANv2
![image](https://user-images.githubusercontent.com/32179857/148708059-dac16ade-52d5-4836-9070-2125fe071bb2.png)

### Proposed framework
- Let X and Y be the sets of images and possible domains, respectively.
- Our goal is to train a single **generator G** that can generate **diverse** images of each domain y that corresponds to the image x. 
- We generate **domain-specific** style vectors in the learned style space of each domain and train G to reflect the style vectors.

#### Generator
- Our generator G translates an input image x into an output image G(x, s) reflecting a domainspecific style code s.
- Style code s is provided either by the mapping network F or by the style encoder E.
- An adaptive instance normalization (AdaIN) is used to inject s into G.
- The authors observe that s is designed to represent a style of a specific domain y, which removes the necessity of providing y to G and allows G to synthesize images of all domains.

#### Mapping network
- Given a latent code z and a domain y, a mapping network F generates a style code s = F<sub>y</sub>(z), where F<sub>y</sub>(·) denotes an output of F corresponding to the domain y.
- F consists of an MLP with multiple output branches to provide style codes for all available domains.
- F can produce diverse style codes by sampling the latent vector z ∈ Z and the domain y ∈ Y randomly.

#### Style encoder
- Given an image x and its corresponding domain y, our encoder E extracts the style code s = E<sub>y</sub>(x) of x. 
- Here, E<sub>y</sub>(·) denotes the output of E corresponding to the domain y.

#### Descriminator
- The discriminator D is a multitask discriminator, which consists of multiple output branches.
- Each branch Dy learns a binary classification determining whether an image x is a real image of its domain y or a fake image G(x, s) produced by G.

### Training objectives
#### Adersarial objectives
- During training, we sample a latent code z ∈ Z and a target domain ỹ ∈ Y randomly, and generate a target style code es = F<sub>ỹ</sub>(z). 
- The generator G takes an image x and es as inputs and learns to generate an output image G(x,![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png)) via an adversarial loss
  ![image](https://user-images.githubusercontent.com/32179857/148708847-d90c7e63-eae6-447d-b14e-83b0563ddfed.png) 
  - where Dy(·) denotes the output of D corresponding to the domain y.
  - The mapping network F learns to provide the style code ![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png) that is likely in the target domain ![image](https://user-images.githubusercontent.com/32179857/148709039-8a174e08-2cf4-4d04-b21f-7fff84e01ba8.png)
  -  G learns to utilize es and generate an image G(x,![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png)) that is indistinguishable from real images of the domain ![image](https://user-images.githubusercontent.com/32179857/148709039-8a174e08-2cf4-4d04-b21f-7fff84e01ba8.png).

#### Style reconstruction
- In order to enforce the generator G to utilize the style code es when generating the image G(x,![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png)), we employ a style reconstruction loss.
  ![image](https://user-images.githubusercontent.com/32179857/148709129-da72fccc-7ab6-479d-ac93-f68e8f438180.png)
- It employes multiple encoders to learn a mapping from an image to its latent code.
- The notable difference from previous works is that we train a single encoder E to encourage diverse outputs for multiple domains.

#### Style diversification
- To further enable the generator G to produce diverse images, we explicitly regularize G with the diversity sensitive loss.
  ![image](https://user-images.githubusercontent.com/32179857/148709284-9e5239ec-5b38-41ee-bbcb-b83ae7ef4ba6.png)
- where the target style codes ![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png)1 and ![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png)2 are produced by F conditioned on two random latent codes z<sub>1</sub> and z<sub>2</sub> 

#### Preserving source characteristics.
- To guarantee that the generated image G(x,![image](https://user-images.githubusercontent.com/32179857/148709029-d2788276-512d-4a62-9a78-bf908c691338.png)) properly preserves the domaininvariant characteristics (e.g. pose) of its input image x,
  we employ the cycle consistency loss
  ![image](https://user-images.githubusercontent.com/32179857/148714730-9e534ca9-8672-411f-8fc0-a503f50d5b21.png)

#### Full objective (total loss)
  ![image](https://user-images.githubusercontent.com/32179857/148714764-fb30b7b9-6718-4eb4-b22e-f7ce1edbb720.png)
- where λ<sub>sty</sub>, λ<sub>ds</sub>, and λ<sub>cyc</sub> are hyperparameters for each
term

&nbsp;

## Experiments
### Datasets
- CelebA-HQ (2 domains)
  - male
  - female
- AFHQ (3 domains)
  - cat
  - dog
  - wildlife
- Learning without supervision (no other characteristic inputs (ex: facial attributes, breed))
- input size: 256x256
### Evaluation metrics
- FID ([Frechét inception distance](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance))
- LPIPS ([learned perceptual image patch similarity](https://arxiv.org/abs/1801.03924))
### Analysis of individual components
![image](https://user-images.githubusercontent.com/32179857/148715380-7c781cbc-573b-4021-afe5-4e0baf7347f6.png)
- configuration(A): StarGAN
- configuration(B): (A) + multi-task discriminator (allow to change global structure of image)
- configuration(C): (B) + R1 regularization + switching the depth-wise concatenation to adaptive instance normalization (AdaIN)
- configuration(D): (C) + directly induce latent code z into Generator G --> has no power of separating domains
- configuration(E): (C) + domain-specific style code s through mapping network, inject into Generator(E)
- configuration(F): (E) + adopting the diversity regularization (at Style diversification section) --> StarGANv2

![image](https://user-images.githubusercontent.com/32179857/148715954-5b78c9bd-5aaf-4a73-8600-b37ed39b6f71.png)

&nbsp;

### Comparison on diverse image synthesis

#### Latent-guided synthesis
![image](https://user-images.githubusercontent.com/32179857/148716023-51e0f4a2-6892-4540-80cc-c14be738278b.png)
![image](https://user-images.githubusercontent.com/32179857/148716218-468fbe7e-bd3b-48ae-b32d-014886d60c93.png)


#### Reference-guided synthesis
![image](https://user-images.githubusercontent.com/32179857/148716113-566b2bec-cb34-4c47-b2cb-96467b2971bf.png)
![image](https://user-images.githubusercontent.com/32179857/148716232-5a6af91b-eff0-49c5-b1ab-3dee0a8ecc30.png)

&nbsp;

## Discussion
### Authors' thought on the reason why StarGAN successes.
- The style code is separately generated per domain by the multi-head mapping network and style encoder.
  - the generator can only focus on using the style code, whose domain-specific information is already taken care of by the mapping network.
- The style space is produced by learned transformations
  - provides more flexibility to our model than the baselines which assume that the style space is a fixed Gaussian distribution.
- Modules benefit from fully exploiting training data from multiple domains

&nbsp;

## Additional
### Training details
![image](https://user-images.githubusercontent.com/32179857/148716460-d9136f82-857a-4a80-8cc8-3114ee6fd3d5.png)


### References
- https://arxiv.org/pdf/1912.01865v2.pdf

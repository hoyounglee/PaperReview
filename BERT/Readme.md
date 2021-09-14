
## BERT
- Bidirectional Encoder Representations from Transformers
- Initially considered as Language Representation, but being used as text classification, QA, and other multiple tasks thesedays.
- Bidirectional method is characteristic
- First model that can be pretrained (Other NLP model had not been difficult to pre-train)


## Method
![image](https://user-images.githubusercontent.com/32179857/133185023-baf53540-1b02-43f1-9078-e52bd7dd4a96.png)

## How it works

### Training

- Encoder from Attention Module

  ![image](https://user-images.githubusercontent.com/32179857/133193432-1594df85-97a9-4739-8873-19243429589b.png) --> ![image](https://user-images.githubusercontent.com/32179857/133195364-b287b002-6775-4ff4-bb21-0c5f2dd3a7a4.png)

#### Positional Encoding
- No convolution, No Recurrence
    
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. (Vaswani et al., Attention Is All You Need, 2017)
    
- It inserts token's relational or absolute positional information which is called **Positional Encoding**. 
    ![image](https://user-images.githubusercontent.com/32179857/133193957-37e08746-5bb7-43a6-a76d-336b35e592c4.png)
#### Input Embedding
- BERT uses input embedding (positional embedding) rather than positional encoding
    ![image](https://user-images.githubusercontent.com/32179857/133194292-c682f7d7-5c60-4e73-bd06-e61792332ea9.png)
- in the codes,
~~~
e = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
~~~
1. Token Embedding
    - Divide sentence using special token (CLS, SEP)
2. Segmentation Embedding 
    - Separate order of sentences
3. Position Embedding
    - Order words(token) to give relationship between each other

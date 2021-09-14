
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
    
#### Positional Encoding
- No convolution, No Recurrence
    
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. (Vaswani et al., Attention Is All You Need, 2017)
    
- It inserts token's relational or absolute positional information which is called **Positional Encoding**. 
    ![image](https://user-images.githubusercontent.com/32179857/133193957-37e08746-5bb7-43a6-a76d-336b35e592c4.png)

### Multi-Head Attention
- calcuate attnetion 'h' times with different weight matrix --> and concat them together.
    ![image](https://user-images.githubusercontent.com/32179857/133196614-3ab386d0-37c4-4a56-85f7-cf7837ef5103.png)
- For BERT-base model, it divides each Token vector 768 dim into 64 x 12 pieces. After then, Scaled Dot-Product Attention is implmented and concat them altogether into 768 dimension again. --> this means 768 dim. vector gets 12 times of attention for each part.

    ![image](https://user-images.githubusercontent.com/32179857/133196932-d89bf94a-9922-48f6-a674-50d6534feba6.png)
- Scaled Dot-Product attention gets Q, K, V as inputs. these are the placeholder for the inputs, 1) results of fully-connection of embedding, 2) previous block's results (like RNN) 
- Transformer deals Q as decoder's hidden state, K as encoder's hidden, and V as normalized weights given attention to K.
- BERT does not use decoder and Q=K=V. --> So it becomes 'Self-Attention'.
- 



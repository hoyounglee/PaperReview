
## BERT
- Bidirectional Encoder Representations from Transformers
- Initially considered as Language Representation, but being used as text classification, QA, and other multiple tasks thesedays.
- Bidirectional method is characteristic
- First model that can be pretrained (Other NLP model had not been difficult to pre-train)


## Method
![image](https://user-images.githubusercontent.com/32179857/133185023-baf53540-1b02-43f1-9078-e52bd7dd4a96.png)

## How it works

- Encoder from Attention Module

  ![image](https://user-images.githubusercontent.com/32179857/133193432-1594df85-97a9-4739-8873-19243429589b.png) --> ![image](https://user-images.githubusercontent.com/32179857/133195364-b287b002-6775-4ff4-bb21-0c5f2dd3a7a4.png)

### Input Embedding
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
    
### Positional Encoding
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
- Q, K, V?

    ![image](https://user-images.githubusercontent.com/32179857/133197432-68fe943a-dcfe-4848-855c-8a291b018582.png)

![image](https://user-images.githubusercontent.com/32179857/133368608-fcd8a5fa-873a-4ecd-a602-7731940db4b4.png)

### Activation - GELU
![image](https://user-images.githubusercontent.com/32179857/133368712-1ba839dc-7f68-4f73-9737-b91c675f61c0.png)


## Training
- we can manipulate the model depending on the purpose of the task.

#### Masked Langauage Model
- Referred from, https://docs.likejazz.com/bert/

>Masked Language Model은 문장의 다음 단어를 예측하는 것이 아니라 문장내 랜덤한 단어를 마스킹하고 이를 예측하도록 하는 방식으로 Word2Vec의 CBOW 모델과 유사하다. 하지만 MLM은 Context 토큰을 Center 토큰이 되도록 학습하고 Weights를 벡터로 갖는 CBOW와 달리, 마스킹된 토큰을 맞추도록 학습한 결과를 직접 벡터로 갖기 때문에 보다 직관적인 방식으로 볼 수 있다. 마스킹은 전체 단어의 15% 정도만 진행하며, 여기에는 재밌게도 모든 토큰을 마스킹 하는게 아니라 80% 정도만 <MASK>로 처리하고 10%는 랜덤한 단어, 나머지 10%는 정상적인 단어를 그대로 둔다.

><MASK> 토큰에 대해서만 학습한다면 Fine-tuning시 이 토큰을 보지 못할 것이고 아무것도 예측할 필요가 없다고 생각해 성능에 영향을 끼칠 것이다. 때문에 <MASK> 토큰이 아닌 것도 예측하도록 학습하여 문장의 모든 단어에 대한 문맥 표현Contextual Representation이 학습되도록 한다.

>Word2Vec의 경우 Softmax의 연산 비용이 높기 때문에 Hierachical Softmax 또는 Negative Sampling을 사용하는데, BERT는 전체 Vocab Size에 대한 Softmax를 모두 계산한다. 구글에서 공개한 영문 학습 모델의 Vocab Size는 30,522개로, Output Size는 Vocab Size와 동일한 갯수의 Linear Transformation 결과의 Softmax를 정답으로 한다. 따라서, Loss는 정답 벡터 위치와 Vocab Size 만큼의 Softmax 차이가 된다. 한편 한글 모델의 경우에는 형태소 분석 결과가 10만개를 넘어가는 경우가 흔하므로 학습에 더욱 오랜 시간이 걸린다.

#### Next Sentence Prediction
- Referred from, https://docs.likejazz.com/bert/
  
>Next Sentence Prediction은 두 문장을 주고 두 번째 문장이 코퍼스 내에서 첫 번째 문장의 바로 다음에 오는지 여부를 예측하도록 하는 방식이다. 이 방식을 사용하는 이유는 BERT는 Transfer Learning으로 사용되고 QA와 Natural Language Inference(NLI)등의 태스크에서는 Masked Language Model로 학습하는 것 만으로는 충분하지 않았기 때문이다. 두 문장이 실제로 이어지는지 여부는 50% 비율로 참인 문장과 랜덤하게 추출되어 거짓인 문장의 비율로 구성되며, [CLS] 벡터의 Binary Classification 결과를 맞추도록 학습한다.
  
#### Embedding
- The biggest characteristic of BERT is 'dynamic embedding'.
- With this, even same word can have different embedding value.
  
```
sentences = [
    '나 는 너 를 사랑 하다 여',
    '나 는 너 를 사랑 하다 였 다',
    '사랑 누 가 말하다 였 나',
]
(q_length, q_tokens, q_embedding, q_ids) = bc.encode(sentences)

love_1 = q_embedding[0][5]
love_2 = q_embedding[1][5]
love_3 = q_embedding[2][1]

spatial.distance.cdist([love_2, love_3], [love_1], metric='cosine')
--
array([[0.0546998 ],
       [0.52740145]])
```
  
  
#### Sentence Representation
- Trial for finding good feature layer to generate good representation
  ![image](https://user-images.githubusercontent.com/32179857/133369750-b2413d75-c51e-483c-921a-398d812435b9.png)

- pooling_layer=-1 => close layer to output
- pooling_layer=-12 => close layer to input
- here, pooling_layer=-12 looked the best as a word embedding
  
  ![image](https://user-images.githubusercontent.com/32179857/133369873-c404b151-5cd6-41bf-be7c-ddd6ae156ff2.png)

#### Named Entity Recognition (NER)
- here, the authors tested feature concatenation, and found that concat does not have remarkable difference.
  ![image](https://user-images.githubusercontent.com/32179857/133369946-c990bd1a-a04c-44f5-9cae-5f99712b0ed2.png)

  
## Performance
  #### 1. GLUE
  
  ![image](https://user-images.githubusercontent.com/32179857/133370201-72d27b2f-7a70-4611-9dcb-4d10c20cd6b5.png)
 
  #### 2. SQuAD v1.1
  
  ![image](https://user-images.githubusercontent.com/32179857/133370232-24cb6fd6-e65d-4551-aded-6c7b4ae90ad8.png)

### example
  ![bert](https://user-images.githubusercontent.com/32179857/133370604-beef53de-d11d-4b54-ac54-deeae9d7c0a8.gif)

  
  ![image](https://user-images.githubusercontent.com/32179857/133370469-4f26fb7c-351d-4380-9748-cd69bd446a90.png)


## References
- https://docs.likejazz.com/bert/#fn:fn-12
- https://hwiyong.tistory.com/392
- https://ebbnflow.tistory.com/151

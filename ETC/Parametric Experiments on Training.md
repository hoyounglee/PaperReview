## Model
- custom model which has simple structure of binary(categorical) classifier.
- Model structure
```python
classifier.add(Conv2D(16, (3, 3), padding='same', input_shape = (180, 180, 3), activation = 'relu'))
classifier.add(Conv2D(16, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25

classifier.add(Conv2D(32, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25

# Adding a second convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25

# Adding a third convolutional layer
classifier.add(Conv2D(64, (3, 3), padding='same', activation = 'relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.5)) # antes era 0.25

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dropout(0.5)) 
classifier.add(Dense(units = 2, activation = 'softmax'))
```

## Dataset
- Simple cats and dogs images are used for the experiments
- variations
    ```
    Train: Found 18727 images belonging to 2 classes.
    Test:  Found 4683 images belonging to 2 classes.
    ```
    
    
## Experiments
- All the experimental figures are displayed by showing intermediate layers' output.
- The result includes **top 12** layers.
- Only 1 test image (herein, a dog image) is used for the experiment for the equal comparison.
- Test Image

   <img src="https://user-images.githubusercontent.com/32179857/137627741-aa07bafe-1d45-4b36-a73e-0f914b521b53.png" width="250" height="250" />


### Batch size
- default hypyer parameter / configurations
    - image scaling  : [0, 1] scaling
    - training epoch : 10
    - input dimension: 180
    
    #### 1. batch size 8
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137630169-d9cad118-f8ca-4394-afdd-e79a72b92d9e.png" width="650" height="250" />
    
    - Intermediate Layer outputs 

    <img src="https://user-images.githubusercontent.com/32179857/137629144-54d0ebcc-b35c-4192-b80f-48d19d80989a.png" width="950" height="750" />

    #### 2. batch size 32
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137629971-045c6f27-4e6d-4922-9406-8a4548623adf.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137629942-c3019a83-feb6-4610-8172-2729fce8b9ff.png" width="950" height="750" />


### Input dimension
- default hypyer parameter / configurations
    - image scaling  : [0, 1] scaling
    - training epoch : 10
    - Batch size.   : 32

    #### 1. input dimension 180
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137629971-045c6f27-4e6d-4922-9406-8a4548623adf.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137629942-c3019a83-feb6-4610-8172-2729fce8b9ff.png" width="950" height="750" />



    #### 2. input dimension 90
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137631212-b9fb5ca3-0c73-4fe7-b4f5-b45f3191e79a.png" width="650" height="250" />    

    - Intermediate Layer outputs

    <img src="https://user-images.githubusercontent.com/32179857/137631634-2beaf04c-8c17-4169-bc6e-01b7fa77039c.png" width="950" height="750" />   

    
### Scaling
- default hypyer parameter / configurations
    - Batch size.    : 32
    - training epoch : 10
    - input dimension: 180

    #### 1. Scaling [0, 1]
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137629971-045c6f27-4e6d-4922-9406-8a4548623adf.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137629942-c3019a83-feb6-4610-8172-2729fce8b9ff.png" width="950" height="750" />

    #### 2. Scaling [0, 255]
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137632248-bf0370ab-4102-4204-b969-a193902b74f1.png" width="650" height="250" />

    - Intermediate Layer outputs
 
    <img src="https://user-images.githubusercontent.com/32179857/137632288-34fa1817-2fe7-4546-91b7-6094e3671631.png" width="950" height="750" />

- Why Scaling improves the training performance?
> 1. Treat all images in the same manner:
>> some images are high pixel range, some are low pixel range. The images are all sharing the same model, weights and learning rate. The high range image tends to create stronger loss while low range create weak loss, the sum of them will all contribute the back propagation update. But for visual understanding, you care about the contour more than how strong is the contrast as long as the contour is reserved. Scaling every images to the same range [0,1] will make images contributes more evenly to the total loss. In other words, a high pixel range cat image has one vote, a low pixel range cat image has one vote, a high pixel range dog image has one vote, a low pixel range dog image has one vote... this is more like what we expect for training a model for dog/cat image classifier. Without scaling, the high pixel range images will have large amount of votes to determine how to update weights. For example, black/white cat image could be higher pixel range than pure black cat image, but it just doesn't mean black/white cat image is more important for training.
> 2. Using typical learning rate: 
>> when we reference learning rate from other's work, we can directly reference to their learning rate if both works do the scaling preprocessing over images data set. Otherwise, higher pixel range image results higher loss and should use smaller learning rate, lower pixel range image will need larger learning rate.

### Optimizer
- default hypyer parameter / configurations
    - Batch size.    : 32
    - training epoch : 10
    - input dimension: 180
    - image scaling  : [0, 1] scaling
    <img src="https://user-images.githubusercontent.com/32179857/137632675-734f0e23-518f-4879-814c-ba86e5005117.png" width="650" height="350" />

    #### 1. RMSProp
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137629971-045c6f27-4e6d-4922-9406-8a4548623adf.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137629942-c3019a83-feb6-4610-8172-2729fce8b9ff.png" width="950" height="750" />
    
    #### 2. SGD
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137632550-e7185a95-022a-4747-93ee-a3dd585a83f7.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137632563-f18e055a-00ec-4d20-9c48-49535679c0fe.png" width="950" height="750" />

    #### 4. Adam
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137634458-111347fd-fae9-44f2-a9da-63273fcc13e2.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137634538-c2c4dce5-c7b6-4995-804d-13b4465440e0.png" width="950" height="750" />
   

### Normalization
- default hypyer parameter / configurations
    - Batch size.    : 32
    - training epoch : 10
    - input dimension: 180
    - image scaling  : [0, 1] scaling
    - optimizer      : RMSProp

    #### 1. No normalization
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137629971-045c6f27-4e6d-4922-9406-8a4548623adf.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137629942-c3019a83-feb6-4610-8172-2729fce8b9ff.png" width="950" height="750" />
    
    #### 2. Batch Norm after activation. (Conv -> ReLU -> Batchnorm)
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137634921-d41d0600-08b6-442e-80c1-9486c80b08f7.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137634963-ffb0ca73-9662-41ab-ac01-3f9aa9f544ba.png" width="950" height="750" />

    
    #### 3. Batch Norm before activation. (Conv -> Batchnorm -> ReLU)
    - Training flow

    <img src="https://user-images.githubusercontent.com/32179857/137635788-9c93e6ae-0452-4f4b-9fff-b326dc1b79d7.png" width="650" height="250" />

    - Intermediaite Layer outputs
    
    <img src="https://user-images.githubusercontent.com/32179857/137635810-bec44422-0eab-4c36-8d02-babcf3eec257.png" width="950" height="750" />
    
    
    

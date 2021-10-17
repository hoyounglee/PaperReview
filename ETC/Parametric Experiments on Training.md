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
    
    #### 1. batch size 16
    - Training flow
    
    <img src="https://user-images.githubusercontent.com/32179857/137628260-fb6719eb-a0db-41e2-b31e-d81ef8460f0e.png" width="650" height="250" />
    
    - Intermediate Layer outputs 

    <img src="https://user-images.githubusercontent.com/32179857/137629144-54d0ebcc-b35c-4192-b80f-48d19d80989a.png" width="950" height="750" />




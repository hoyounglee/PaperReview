# Convolutions
## basic convolution
![conv1](https://user-images.githubusercontent.com/6396598/133370621-817d9a60-130d-4d5d-8bb2-696b742bb6d9.gif)

### how to colculate? 
```
1. input size: 
  - width: W1 = 5
  - height: H1 = 5
  - Dimension: D1 = 3
2. convolution layer hyper parameter
  - Number of filters K = 2,
  - their spatial extent F = 3,
  - the stride S = 2,
  - the amount of zero padding P = 1
3. output size:
  - width: W2 = (W1−F+2P)/S+1 = (5 - 3 + 2)/2  + 1 = 3
  - height: H2=(H1−F+2P)/S+1 = (5 - 3 + 2)/2 + 1 = 3
  - Dimension: D2 = K = 2
```
![image](https://user-images.githubusercontent.com/6396598/133527409-eb1be1ea-547d-4fcc-9f83-51a4647a50d1.png)

## Dilated convolution
![conv2](https://user-images.githubusercontent.com/6396598/133370681-96be04ad-a4c5-40aa-95d0-e5edda282951.gif)

## Transposed convolution
![conv3](https://user-images.githubusercontent.com/6396598/133370834-15d11349-64f0-414f-b5c9-8c2b626aaee5.gif) ![conv4](https://user-images.githubusercontent.com/6396598/133370851-8b320875-9ca2-497e-82fe-ddd5101b404f.gif)

## Seperable convolution
![image](https://user-images.githubusercontent.com/6396598/133370928-ea0dde31-b4cd-4d11-b99c-d10f59139239.png)

## Depthwise convolution
![image](https://user-images.githubusercontent.com/6396598/133370937-ecf5c65a-0c9b-4330-8c74-d56c2f86a254.png)

## Depthwise separable convolution
![image](https://user-images.githubusercontent.com/6396598/133370947-43147e00-6fa9-475b-8e64-47843b0d458c.png)

## Pointwise convolution
![image](https://user-images.githubusercontent.com/6396598/133370960-04b2f786-77fb-4f00-bcfa-e25aba5a1bd2.png)

## Grouped convolution
![image](https://user-images.githubusercontent.com/6396598/133370975-c437bb1c-613a-4c59-831b-55b2e5f0ff8d.png)

## Deformable convolution
![image](https://user-images.githubusercontent.com/6396598/133370991-11936fe7-e7b5-4579-a732-ff4e5bfa53b7.png)

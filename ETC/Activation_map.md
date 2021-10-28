# Activation Map

| Method  | What it does |
| ------------- | ------------- |
| GradCAM[[ref]](https://arxiv.org/abs/1610.02391)  | Weight the 2D activations by the average gradient |
| GradCAM++[[ref]](https://arxiv.org/abs/1710.11063)  | Like GradCAM but uses second order gradients |
| XGradCAM  | Like GradCAM but scale the gradients by the normalized activations |
| AblationCAM[[ref]](https://ieeexplore.ieee.org/abstract/document/9093360/)  | Zero out activations and measure how the output drops |
| ScoreCAM[[ref]](https://arxiv.org/abs/1910.01279)  | Perbutate the image by the scaled activations and measure how the output drops |
| EigenCAM[[ref]](https://arxiv.org/abs/2008.00299)  | Takes the first principle component of the 2D Activations |
| EigenGradCAM  | Like EigenCAM but with class discrimination |
| LayerCAM[[ref]](http://mftp.mmcheng.net/Papers/21TIP_LayerCAM.pdf)  | Spatially weight the activations by positive gradients. Works better especially in lower layers |
| FullGrad[[ref]](https://arxiv.org/abs/1905.00780)  | Computes the gradients of the biases from all over the network, and then sums them |


# CS 415 Computer Vision Homework 2
## Hal Brynteson 

1. Exercise 5.8
    * a) **Are convolutions translation invariant?**
        Convolutions are not translation invariant. The output image of the convolution operation will be different if that image has been translated. 
    * b) **Are convolutions translation equivariant?**
        Convolutions are translation equivariant.
    * c) **Are convolutions scale invariant?**
        Convolutions are not scale invariant. 
    * d) **Are convolutions scale equivariant?**
    * See Table for reasoning: 
        In this example, the convolution operation is a gaussian blur kernel. 
        ![Invariance/Equivariance Table](q1/table.png)

2. Evaluating Invariance Properties of HCD
    * a) Detect Corners: For each image, the block size, ksize, k and threshold were set to get less than 1000 corners where most of the corners detected appear close to a feature. 
    * b) HOG features: 
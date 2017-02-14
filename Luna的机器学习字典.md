## A

* auxillary information

  辅助信息

## B

* Back propagation

  Please use the keyword deriviate to get more info for specific derivatives of each nonlinear activation function and softmax function.

* batch normalization
  把一个batch的x都变成$\frac{x-\mu}{\sigma}$ 

  ​

* Bayesian parameter estimation 贝叶斯参数估计 (BPE)
  $$
  p(\theta|X)=\frac{p(X|\theta)p(\theta)}{\int{p(X|\theta)p(\theta)}~d\theta}
  $$
  ​

## C

* convolution related concepts

  参见[演示](https://github.com/vdumoulin/conv_arithmetic)

  * padding

    在图片周围添0，分arbitrary padding, half padding, full padding

  * strides
    步幅不是1，而是2啊，3啊

    ​

## D

* deconvolution

  本义是conv的逆运算，但实际是transposed convolution的误用，参见此词条

  ​

* distributed representation

  词向量

  ​

## E

* encoder decoder structure

  Language A => RNN encoder => hidden state S => RNN decoder => Language B

  ​


* estimation of probability distribution (density)

  分成参数估计和非参数估计

  参数估计是预先假设每一个类别的概率密度函数的形式已知，而具体的参数未知，常见有最大似然估计maximum likelihood estimation(MLE) 和贝叶斯估计

  非参数估计

  ​

* exponential moving average

  有一个momentum $m$

  $\text {mean}_{t+1}=(1-m) \times \text {mean}_{t} + m \times \text {batch_mean}_{t+1}$

## F

## G

## H

## I

## J

## K

* kernel
  * uinversal kernel/charasteristic kernel：能把空间填满的kernel

* k nearest neighbor 

  k是hyperparameter

  随着k增加，参考范围变大，variance变小，bias变大

  弱点，dimensionality: 维度一多就歇菜了

  * irrelevant attributes，有很多维度的信息没有用
  * lack of similarity 维度高了之后，就很难“邻近”了
## L

* leaky ReLU

  ReLu是$\max(0,x)$ ，即x为负的时候，gradient是0

  leaky ReLU 是 x为负的时候，gradient是 $\lambda$


## M

*  maximum likelihood estimation 最大似然估计 MLE
   $$
   \hat \theta = \underset{\theta}{\text {argmax}}\prod_{i=1}^n p(X_i|\theta)
   $$
   ​

   一种对分布的参数估计方法，即模型已定，参数未知，求参数

   求解过程：

   1. 写出似然函数；
   2. 对似然函数取对数，并整理；
   3. 求导数 ；
   4. 解似然方程

   另一种是非参数估计，回避了对分布的假设。很多情况下我们没有分布以及条件分布（label）信息

   ​

*  model

   *  architecture / decision rule
   *  loss function
   *  gradient of loss
   *  update rule

   ​

## N

* nearest neighbor

  参加K nearest neighbor

  Advantages:

  * 简单
  * 错误率很低 only 2 times error-prone as the best imaginable classifier
  * 天生非线性

## O

## P

* parzen window

  一种非参数估计方法，又叫核密度估计（kernel density estimation）是在概率论中用来估计未知的密度函数的
  $$
  f(x)=\frac{1}{nh}\sum_{i=1}^nK(\frac{x-x_i}{h})
  $$
  其中，$K(\cdot)$ 是个kernel，核函数，$h$ 叫bandwidth，带宽，常用$h=\frac{1}{\sqrt n}$ ，$x$ 是mean

  ​

* perceptron

  a **linear classifier** on top of a simple **feature extractor**
  $$
  y=sign(\sum_{i=1}^nW_iF_i(X)+b)
  $$
  其中feature需要自己设计


* polynomial mapping

  construct new **feature vectors**: 
  $$
  \Phi(1,x_1,x_2)=(1,x_1,x_2,x_1^2,x_2^2,x_1x_2)
  $$

* Pooling methods

  * average + non-linearity
  * max
  * log-sum-exp $y=\frac{1}{\beta}log[\frac{1}{n}\sum_ie^{\beta x_i}]$
  * Lp $(\sum_i x_i^p)^\frac{1}{p}$
  * Sort pooling
  * variance 
## Q

## R

* ReLU

  * Function: $y=\max(x, 0)$

  * Derivative: 

  * $$
    \frac{\partial y}{\partial x} = \begin{cases} 
    						1, & \text{if $x>0$}.\\
       						    0, & \text{if $x \leq 0$}.
    					    \end{cases}
    $$





## S

* Sigmoid function

  * Function: $y=\frac{1}{1+e^{-x}}$


*   Derivative: $\frac{\partial y}{\partial x}=\frac{1}{1+e^{-x}} - (\frac{1}{1+e^{-x}})^2$

    ​

*   Softmax

    * Convert evidence to probability

    * Input: vector $X=(x_1, x_2, …, x_m)$ each $x_i$ is a evidence for a circumstance 

    * Output: vector $Y = (y_1, y_2, …, y_m)$ each $y_i$ is a probability for a circumstance

    * Function: $P(y=i|X) = y_i = \frac{e^{x_i}}{\sum_{j=1}^de^{x_j}}$ 

    * Derivative: 

    * $$
      \frac{\partial y_i}{\partial x_k} = \begin{cases}
      						     \frac{e^{x_i} \cdot \sum - e^{x_i} }{\sum^2} =y_i \cdot (1-y_i),~~~~ i=k \\
      						    \frac{- e^{x_i} \cdot e^{x_k} }{\sum^2}=-y_i \cdot y_k, ~~~~i\neq k \\ 
      					                 
      						 \end{cases}
      $$

    * ​

*   Stochastic Gradient Descent (SGD)

        weight = weight - learning_rate * gradient

## T

* Tanh

  * Function: 
    $$
    y=\frac{e^{2x}-1}{e^{2x}+1} ~~\text{or}\\
    y=\frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}~~\text{or}\\
    y=\frac{2}{1+e^{-2x}}-1 ~~\text{or}\\
    y=\frac{\sinh(x)}{\cosh(x)} =\tanh(x)
    $$

  * Derivative: 

  * $$
    \frac{\partial y}{\partial x}=\frac{\cosh^2(x) - \sinh^2(x)}{\cosh^2(x)}\\
    =1-\frac{\sinh^2(x)}{\cosh^2(x)}\\
    =1-\tanh^2(x)
    $$

    ​

* training a model

  A typical training procedure for a neural network is as follows:

  * define the neural network that has some learnable parameters (or weights)

  * iterate over a dataset of inputs:

    * process input through network
    * compute the loss (how far is the output from being correct)

    * propagate gradients back into the network's parameters

    * update the weights of the network

      * typically using a simple update rule: weight = weight + learning_rate * gradient

        ​

* transposed convolution

  又名backward convolution, fractally strided convolution, upsampling convolution.

  ![](no_padding_strides_transposed.gif)

  ​

## U

## V

## W

## X

## Y

## Z




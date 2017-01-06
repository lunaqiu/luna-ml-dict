## A

## B

* batch normalization
  把一个batch的x都变成$\frac{x-\mu}{\sigma}$ 

## D

## E

* Estimation of probability distribution (density)

  分成参数估计和非参数估计

  参数估计是预先假设每一个类别的概率密度函数的形式已知，而具体的参数未知，常见有最大似然估计maximum likelihood estimation(MLE) 和贝叶斯估计

  非参数估计

* exponential moving average

  有一个momentum $m$

  $\text {mean}_{t+1}=(1-m) \times \text {mean}_{t} + m \times \text {batch_mean}_{t+1}$

## F

## G

## H

## I

## J

## K

## L

* leaky ReLU

  ReLu是$\max(0,x)$ ，即x为负的时候，gradient是0

  leaky ReLU 是 x为负的时候，gradient是 $\lambda$


## M

*  maximum likelihood estimation 最大似然估计 MLE

   一种对分布的参数估计方法，即模型已定，参数未知，求参数

   求解过程：

   1. 写出似然函数；
   2. 对似然函数取对数，并整理；
   3. 求导数 ；
   4. 解似然方程

## N

## O

## P

* parzen window

  一种非参数估计方法，又叫核密度估计（kernel density estimation）是在概率论中用来估计未知的密度函数的
  $$
  f(x)=\frac{1}{nh}\sum_{i=1}^nK(\frac{x-x_i}{h})
  $$
  其中，$K(\cdot)$ 是个kernel，核函数，$h$ 叫bandwidth，带宽

## Q

## R

## S

## T

## U

## V

## W

## X

## Y

## Z




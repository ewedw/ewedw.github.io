---
layout: post
title:  Texture Synthesis and Style Transfer
date: 2022-09-14 02:00:00 +0800
permalink: /post/style/

working: true
hide: true
---

# Texture Synthesis and Style Transfer

Here we illustrate a **parametric approach** for **texture synthesis** and then **image style transfer** using *Deep Convolutional Neural Network (CNN)* and [**Gram matrix**](\post\gram){:target="_blank"}.

Paper:

* [Texture Synthesis](https://arxiv.org/abs/1505.07376){:target="_blank"} ([Page](http://www.bethgelab.org/deeptextures/){:target="_blank"})

* [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf){:target="_blank"}

* [Real-Time Style Transfer](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf){:target="_blank"}

* [Learned Representation for Artistic Style](https://arxiv.org/pdf/1610.07629.pdf){:target="_blank"}

* [Controlling Perceptual Factors in Neural Style Transfer](https://arxiv.org/pdf/1611.07865.pdf){:target="_blank"}

### Correlation between feature maps

Consider we have a texture image and want to synthesis a different image with the same texture. In other words, we want an image with different *content* BUT the same *texture*.

A normal classification *CNN* captures the *content* of image in the feature maps in each layer and determine which class the image belongs to. The higher (deeper) the layer, the higher-level content is captured.

BUT how to capture the *texture* of the image?

* *Texture* contains certain relationships between *content*, which are already captured in the learnt features maps.

* To capture the correlation between feature maps in the same layer, [**Gram matrix**](\post\gram){:target="_blank"} is proposed and used.

Consider the following architecture,

![texture_synth](\assets\img\texture_synthesis.png "texture synthesis")

First, we **pretrained** a *CNN* (i.e. VGG-19).  
In the above image, there are 16 **convolutional layers** (grouped as 2-2-4-4-4) and 5 **max-pooling layers**. There are NO **fully-connected layers** at the end. 

* **Filter size** as $$3 \times 3 \times k$$ while **stride** and **padding** both equals 1.

* **Max-pooling** in non-overlapping $$2 \times 2$$ region.

Then, we input the vectorized texture image $$\vec{x}$$ into the trained *CNN*, where each (convolution) layer $$l$$ produces feature maps matrix $$F^l$$ with shape $$N_l \times H_l \times W_l$$ and the feature map for the $$i$$-th filter as $$F^l_i$$ (colored in blue).

$$
\begin{align*}
F^l &\in \mathbb{R}^{N_l \times H_l \times W_l} \\
F^l_i &\in \mathbb{R}^{H_l \times W_l}
\end{align*}$$

![before_vec](\assets\img\cnn_before_vec.png "Feature maps matrix Before vectorized")

We can view it as $$N_l$$ numbers of feature maps $$F^l_i$$. To find the correlation between each feature map $$F^l_i$$, we vectorize them from shape $$H_l \times W_l$$ to shape $$H_l W_l$$ (colored in blue), 

$$
\begin{align*}
F^l &\in \mathbb{R}^{N_l \times H_l W_l} \\
F^l_i &\in \mathbb{R}^{H_l W_l}
\end{align*}
$$

![after_vec](\assets\img\cnn_after_vec.png "Feature maps matrix After vectorized")

and compute the [**Gram matrix**](\post\gram){:target="_blank"} $$G^l \in \mathbb{R}^{N_l \times N_l}$$ with the **inner product** of the vectorized feature maps $$F^l_i$$.

$$G^l = 
\begin{bmatrix}
\vec{F^l_1} \cdot \vec{F^l_1} & \cdots &  \vec{F^l_1} \cdot \vec{F^l_n}\\
\vdots & \ddots & \vdots \\
\vec{F^l_n} \cdot \vec{F^l_1} & \cdots & \vec{F^l_n} \cdot \vec{F^l_n}
\end{bmatrix}$$

where element in row $$i$$ and column $$j$$ is taken as the **dot product**, and $$k$$ denotes the $$k$$-th element in any vectorized feature map $$\vec{F^l_i}$$ or $$\vec{F^l_j}$$, 

$$
\begin{align*}
G^l_{ij} &= \vec{F^l_i} \cdot \vec{F^l_j} \\
&= \sum_k F^l_{ik} \, F^l_{jk}
\end{align*}
$$

A set of **Gram matrices** $$\{ G^1, \dots, G^L\}$$ are produced for $$L$$ layers $$\{ 1, \dots, L\}$$.

### Texture Synthesis with new images

After quantifying *texture* with **Gram matrix**, we are going to generate new images with same texture.

![texture_synth](\assets\img\texture_synthesis.png "texture synthesis")

The approach is as follows,

* We initialize a new image $$\hat{\vec{x}}$$ with noise.

* We feed it to the same *CNN* as above.

* We calculate a set of **Gram matrices** $$\{ \hat{G^1}, \dots, \hat{G^L}\}$$ for $$L$$ layers $$\{ 1, \dots, L\}$$ from feature maps $$\hat{F}^l$$ as above.

* We compare the difference between 

	* the **Gram matrices** $$\{ G^1, \dots, G^L\}$$ for original texture image $$\vec{x}$$

	* and the **Gram matrices** $$\{ \hat{G^1}, \dots, \hat{G^L}\}$$ for generated image $$\hat{\vec{x}}$$

* We do **gradient descent** and [**back-propagation**](https://youtu.be/tIeHLnjs5U8){:target="_blank"} to minimize the difference (match the *texture* of the generated image $$\hat{\vec{x}}$$ to the real texture image $$\vec{x}$$)

There are two things left to be considered, the **loss function** for comparing the difference and the **gradient** of it.

### Loss function

Recall a few notation

* $$\vec{x}$$: the original texture image

* $$\hat{\vec{x}}$$: the generated image  
(We add $$\hat{\,}$$ for the following if it is related to $$\hat{\vec{x}}$$)

* $$\vec{F^l}$$: the vectroized feature maps matrix for layer $$l$$ with shape $$N_l \times H_l  W_l$$

* $$\vec{F^l_i}$$: the vectorized feature maps in $$i$$-th filter for layer $$l$$ with shape $$H_l W_l$$

* $$G^l$$: the **Gram matrix** for layer $$l$$

* $$G^l_{ij}$$: the element in the **Gram matrix** with row $$i$$ and column $$j$$

The layer loss $$E_l$$ for layer $$l$$ is defined as

$$E_l = \frac{1}{4 N_l^2 (H_l W_l)^2} \sum_{i,j} \left( G^l_{ij} - \hat{G^l_{ij}} \right)^2$$

where we are simply taking the difference between the elements in $$G^l$$ and $$\hat{G^l}$$ and calculate the L2 loss.

The **total loss** for all $$L$$ layers $$\{ 1, \dots, L\}$$ is defined as

$$\mathcal{L}(\vec{x}, \hat{\vec{x}}) = \sum_{i=0}^L w_l E_l$$

where $$w_l$$ is the weighting factors of the contribution of each layer $$l$$ to the total loss.

### Gradient

$$
\frac{\partial E_l}{\partial \hat{\vec{F^l_{i}}} } = 
\begin{cases} 
\frac{1}{N_l^2 (H_l W_l)^2} \left( \left( \hat{\vec{F^l_j}} \right)^T \left( G^l_{ij} - \hat{G^l_{ij}} \right) \right) & \text{if } \hat{F^l_{ik}} > 0  \\ 
0 & \text{if } \hat{F^l_{ik}} < 0 
\end{cases}
$$

where $$k$$ denotes the $$k$$-th element in the vectorized feature map $$\hat{\vec{F^l_i}}$$

$$\displaystyle \frac{\partial \mathcal{L}}{\partial \hat{\vec{x}}}$$ can be computed using [**back-propagation**](https://youtu.be/tIeHLnjs5U8){:target="_blank"}. 

### Results

Image from the [paper](https://arxiv.org/abs/1505.07376){:target="_blank"}.

![results](\assets\img\texture_synthesis_results.png "Results")

The labels "conv1_1", "pool1", "pool2", "pool3", "pool4" describe the layers included up to in the calculation of $$\mathcal{L}(\vec{x}, \hat{\vec{x}})$$.

Visit the [page](http://www.bethgelab.org/deeptextures/){:target="_blank"} for more synthesized results.

* The higher (deeper) the layers are up to, the better the quality of the generated image $$\hat{\vec{x}}$$

* A lot of parameters are needed

* Good for more random texture  
(i.e. a bunch of stones)  

* Bad for structured / grid-like texture  
(i.e. bricks wall)  

It is understandable, since the [**loss function**](#loss_function) matches the *texture* (**Gram matrix**) but not the *content* (feature maps). Maybe adding a match on the high-level features (in the higher layers) to the **loss function** can help improve this.

### Non-parametric approach for structured texture

Let's think about it. For structured texture, it is intuitively better to use a [**non-parametric approach**](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf){:target="_blank"}. Given a structured texture (see the bamboo pattern image below), we can simply "copy and paste" a patch from it and generate new images from it. It will look all fine!

![structured bamboo](\assets\img\structured_pattern.png "Structured bamboo pattern")

We are not going to go in-depth for this approach here. The basic idea for **non-parametric approach** for texture synthesis is to generate the new image pixel by pixel, sample from the **nearest neighbours** of the original texture image.

![non-param](\assets\img\texture_synthesis_non_param.png "Non-parametric approach")

View this [paper](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-iccv99.pdf){:target="_blank"} for details.

![results non-param](\assets\img\texture_synthesis_results_non_param.png "Results for non-parametric approach")

### Style Transfer

#TODO

### Advances

#TODO (2017 Texture Review)
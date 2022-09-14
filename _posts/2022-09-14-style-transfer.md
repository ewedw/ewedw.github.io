---
layout: post
title:  Texture Synthesis and Style Transfer
date: 2022-09-14 02:00:00 +0800
permalink: /post/style/

working: true
---

# Texture Synthesis and Style Transfer

Here we illustrate a parametric approach for **texture synthesis** and **image style transfer** using *Deep Convolutional Neural Network (CNN)* and [**Gram matrix**](\post\gram){:target="_blank"}.

Paper:

* [Texture Synthesis](https://arxiv.org/abs/1505.07376){:target="_blank"}

* [Image Style Transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf){:target="_blank"}

### Texture Synthesis

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

Then, we input the vectorized texture image $$\vec{x}$$ into the trained *CNN*, where each (convolution) layer $$l$$ produces feature maps matrix $$F^l$$ with shape $$N_l \times H_l \times W_l$$

$$F^l \in \mathbb{R}^{N_l \times H_l \times W_l}$$

![before_vec](\assets\img\cnn_before_vec.png "Feature maps matrix Before vectorized")

We can view it as $$N_l$$ numbers of feature maps, where the feature map wth $$i$$-th filter denote as $$F^l_i$$. To find the correlation between each feature map $$F^l_i$$, we vectorize them from shape $$H_l \times W_l$$ to shape $$H_l W_l$$, 

$$
\begin{align*}
F^l &\in \mathbb{R}^{N_l \times H_l W_l}
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

where element in row $$i$$ and column $$j$$ is taken as the **dot product**,

$$
\begin{align*}
G^l_{ij} &= \vec{F^l_i} \cdot \vec{F^l_j} \\
&= \sum_k F^l_{ik} \, F^l_{jk}
\end{align*}
$$

and $$k$$ denotes the $$k$$-th element in any vectorized feature map $$F^l_i$$ or $$F^l_j$$.
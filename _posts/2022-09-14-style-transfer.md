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

Consider we have a texture image. We want to synthesis a differen image that has the same texture. In other words, we want an image with different *content* BUT the same *texture*.

A normal classification *CNN* captures the *content* of image in each layer and determine which class the image belongs to. The higher (deeper) the layer, the higher-level content is captured.

BUT how to capture the *texture* of the image?

* *Texture* contain certain relationships between *content*, which are already captured in the learnt features maps.

* To capture the correlation between feature maps in the same layer, [**Gram matrix**](\post\gram){:target="_blank"} is proposed and used.

Consider the following architecture,

![texture_synth](\assets\img\texture_synthesis.png "texture synthesis")

* First, we **pretrained** a *CNN* (i.e. VGG-19).  
In the image, there are 16 *convolutional layers* (grouped as 2-2-4-4-4) and 5 *max-pooling layers*. There are no *fully-connected layers* at the end. 

	* Filter size as $$3 \times 3 \times k$$ while *stride* and *padding* both equals 1.

	* Max-pooling in non-overlapping $$2 \times 2$$ region.

* Then, we input the vectorized texture image $$\vec{x}$$ into the trained *CNN*, where each (convolution) layer $$l$$ produces $$N_l$$ feature maps $$F^l$$ with shape $$H_l \times W_l$$, so the output dimension as

$$N_l \times H_l \times W_l$$

![before_vec](\assets\img\cnn_before_vec.png "Before vectorized")

* To find the correlation between feature maps $$F^l$$ in layer $$l$$, we can first vectorized each feature maps $$F^l \in \mathbb{R}^{H_l \times W_l} \to \mathbb{R}^{H_l W_l}$$ and compute the [**Gram matrix**](\post\gram){:target="_blank"} with the vectorized feature maps $$F^l \in \mathbb{R}^{H_l W_l}$$.

![after_vec](\assets\img\cnn_after_vec.png "After vectorized")
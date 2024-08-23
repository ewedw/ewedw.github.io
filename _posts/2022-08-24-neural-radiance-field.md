---
layout: post
title:  Neural Radiance Field
date: 2022-08-24 04:00:00 +0800
permalink: /post/nerf/

working: true
hide: true
---

# Neural Radiance Field

This is a post based on a brilliant research [NeRF](https://www.matthewtancik.com/nerf){:target="_blank"}.

### Representing a scene with 5D continuous function

For any 3D location $${\color{dodgerblue} \hat{x}}$$, we can represent it using $${\color{dodgerblue} \hat{x}=(x,y,z)}$$.

For any 2D viewing direction $${\color{red} \hat{d}}$$, we can represent it using $${\color{red} \hat{d}=(\theta,\phi)}$$.

![spherical_coord](\assets\img\spherical.png "Spherical coordinates")

Now, let's introduce a continuous function $$F$$ that takes in 5D vector-valued input of $$( {\color{dodgerblue}x,y,z},{\color{red}\theta,\phi})$$, or simply $$({\color{dodgerblue} \hat{x}}, {\color{red} \hat{d}})$$, and output an emitted color $$\textbf{c} = (r,g,b)$$ and **volume density** $$\sigma$$.

Essentially, the function $$F$$ maps $$(\hat{x}, \hat{d}) \to (\textbf{c}, \sigma)$$. We then use a **deep neural network**, or called **multilayer perceptrons (MLP)**, to learn the function $$F$$ with weights $$\Theta$$, denote as $$F_\Theta$$. 

![NeRF-function](\assets\img\NeRF-function.png "5D continuous function")

### Basic Neural Network Architecture

In practice, we will use a *3D Cartesian coordinate* $$(x,y,z)$$ to represent the **unit vector** $$\hat{d}$$, instead of 2D angles $$(\theta,\phi)$$.

Also, to enourage multiview consistency, the *volume density* $$\sigma$$ is a function of ONLY the location $$\hat{x}$$, while the emitted color $$\textbf{c} = (r,g,b)$$ is a function of BOTH the location $$\hat{x}$$ and the viewing direction $$\hat{d}$$.

![nn-arch](\assets\img\nn-architecture.png "Neural Network Architecture")

### Volume Rendering with Radiance Fields


---
layout: post
title:  Spatial Pyramid Matching 
date: 2022-09-13 21:00:00 +0800
permalink: /post/spm/

working: true
hide: true
---

# Spatial Pyramid Matching 

We will start from the basic idea in [**Pyramid Matching Kernel**](https://www.jmlr.org/papers/volume8/grauman07a/grauman07a.pdf){:target="_blank"}.

### Pyramid Matching Kernel

Consider we have two 1-dimensional feature sets $$y$$ and $$z$$ (you can treat them as feature vectors), 

![pyramid_mathcing_kernel](\assets\img\pyramid_matching_kernel.png "pyramid matching kernel")

* In (a), it shows the whole idea. 

	* We first distribute and align the two 1D feature sets vertically. 

	* Then we divide the space into some number of histogram bins, from finer bins (upper-most image) to coarser bins (lower-most image). 

	* We seek out matching points that fall into the same bin. Dotted lines imply new matching points and bold lines imply previously-identified matching points in the finer bins.

* In (b), we display the above finer to coarser bins (without matching) using horizontal histograms, denote as $$H_i(y)$$ and $$H_i(z)$$. 

* We may not need to stop at $$i=2$$ as shown in the above image. We can denote the set of histograms from finer bins to coarser bins for a particular feature $$X$$ as

$$\Psi (X) = \{ H_0(X), \dots, H_i(X) \}$$

* In (c), we display the *intersection* found in (b) using 

$$I_i = I_i (H_i(y), H_i(z)) = \min (H_i(y), H_i(z))$$

* In (d), we find the new matching points in current bin $$N_i$$ by subtracting from the previous layers (where $$I_{-1} = 0$$)

$$N_i = I_i - I_{i-1}$$ 

* Since we are dividing the vectors into bins, we want to have a "bin weight" $$w_i$$ to describe the "matching power" (how strong the correlation is for the matching). $$w_i$$ is *inversely proportional* to the "width of the bin", or called the **side length** in the paper. We normally choose **side length** as the power of $$2$$, i.e. $$2^i$$

$$w_i = \frac{1}{2^i}$$

* The set of weights $$\mathcal{W}$$ as

$$
\begin{align*}
\mathcal{W} &= \left\{ 1, \frac12, \frac14, \dots \right\} \\
&= \left\{ \frac{1}{2^0}, \frac{1}{2^1}, \frac{1}{2^2}, \dots \right\} \\
\end{align*}
$$

* The (unnormalzied) **pyramid match** $$P_\Delta$$ is defined as the sum over all $$N_i$$ with weights $$w_i$$ to find ,

$$P_\Delta = \sum_i w_i N_i$$

* If we expand $$P_\Delta$$, we have

$$P_\Delta = \sum_{i=0}^{L-1} w_i ( \min (H_i(y), H_i(z)) - \min (H_{i-1}(y), H_{i-1}(z)) )$$

where $$L-1 = \lceil \log_2 D \rceil$$ and $$D$$ is the diameter.

### Where comes the diameter $$D$$?

In the above example, we are considering some 1-dimensional vectors and $$D$$ is actually the maximal distance between the value of elements in $$y$$ and $$z$$. In the above image, it is the distance between the upper-most blue dots in $$y$$ and the lower-most red triangles in $$z$$).

If we have some $$d$$-dimensional vectors in feature space $$\mathcal{F}$$, the value of the element in vectors in $$\mathcal{F}$$ has a maximal range $$D$$. We can view the vectors bounded by a $$d$$-dimensional sphere with *diameter* $$D$$.

Thus, the kernel "bins" happens in a spherical ways, and $$H_i(X)$$ has dimension $$r_i$$ as

$$r_i= \left( \frac{D}{2^i} \right)^d$$

### Spatial Pyramid Matching

#TODO Image idea

### Optimal Partial Matching

### Classification and Regression with the Pyramid Match


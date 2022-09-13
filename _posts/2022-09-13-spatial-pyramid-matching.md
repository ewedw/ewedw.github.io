---
layout: post
title:  Spatial Pyramid Matching 
date: 2022-09-13 21:00:00 +0800
permalink: /post/spp/

working: true
---

# Spatial Pyramid Matching 

We will start from the basic idea in [**Pyramid Matching Kernel**](https://www.jmlr.org/papers/volume8/grauman07a/grauman07a.pdf){:target="_blank"}, and then build it into a spatial one.

### Pyramid Matching Kernel

Consider we have two 1D feature sets $$y$$ and $$z$$ (you can treat them as feature vectors), 

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

* We then sum over all $$N_i$$ with weights $$w_i$$ inversely proportional to the "bin width" (actually called the **side length**) to find the **pyramid match** $$P_\Delta$$,

$$
\begin{align*}
w_i &= \{ 1, \frac12, \frac14, \dots \} \\
&= \{ \frac{1}{2^0}, \frac{1}{2^1}, \frac{1}{2^2}, \dots \} \\
P_\Delta &= \sum_i w_i N_i
\end{align*}
$$

* If we expand the **pyramid match kernel**, we have

$$P_\Delta = \sum_{i=0}^{L-1} w_i ( \min (H_i(y), H_i(z)) - \min (H_{i-1}(y), H_{i-1}(z)) )$$

where $$L-1 = \lceil \log_2 D \rceil$$ and $$D$$ is the diameter.

### Why do we have the diameter $$D$$?

In the above example, we are considering some 1D vectors. If we have some $$d$$-dimensional vectors in feature space $$\mathcal{F}$$, the value of the element in vectors in $$\mathcal{F}$$ has a maximal range $$D$$. We can treat $$D$$ as the *diameter* of a $$d$$-dimensinal sphere.

Thus, the kernel "bins" happens in a spherical ways, with radius $$r_i$$ as

$$r_i= \left( \frac{D}{2^i} \right)^d$$

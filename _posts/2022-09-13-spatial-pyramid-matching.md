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

Consider we have two 1D feature sets $$y$$ an $$z$$ (you can treat them as feature vectors), 

![pyramid_mathcing_kernel](\assets\img\pyramid_matching_kernel.png "pyramid matching kernel")

* In (a), it shows the whole idea. We first distribute and align the two 1D feature sets vertically. Then we divide the space into some number of histogram bins, from finer bins (upper-most image) to coarse bins (lower-most image). We seek out macthing points that fall into the same bin. Dotted lines imply new matching points and bold lines imply previously-identified matching points in the finer bins.

* In (b), we display the above finer to coarse bins (without matching) using horizontal historgrams by $$H_i(y)$$ and $$H_i(z)$$.

* In (c), we display the intersection found in (b) using 

$$I_i = \min (H_i(y), H_i(z))$$

* In (d), we find the new matching points in current bin $$N_i$$ by subtracting from the previous layers

$$N_i = I_i - I_{i-1}$$ 
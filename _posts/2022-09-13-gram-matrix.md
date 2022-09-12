---
layout: post
title:  Gram Matrix
date: 2022-09-13 02:00:00 +0800
permalink: /post/gram/
---

# Gram Matrix

### Revising Dot Product

Let's consider a Cartesian 2D plane with *unit basis vectors* $$\hat{i} = (1,0)$$ and $$\hat{j} = (0,1)$$, we can express any vector $$\vec{v}$$ on that plane with

$$
\begin{align*} 
\vec{x} =  \begin{bmatrix} \alpha_1  \\ \alpha_2 \\ \end{bmatrix}  
&= \alpha_1  \begin{bmatrix} 1  \\ 0 \\ \end{bmatrix}  + 
\alpha_2 \begin{bmatrix} 0  \\ 1 \\ \end{bmatrix} \\  
&= \alpha_1 \hat{i} + \alpha_2 \hat{j} 
\end{align*}
$$

Let another vector $$\vec{w} = \beta_1 \hat{i} + \beta_2 \hat{j}$$. When we take the **dot product** of $$\vec{v}$$ and $$\vec{w}$$,

$$\vec{v} \cdot \vec{w} = 
(\alpha_1 \hat{i} + \alpha_2 \hat{j}) \cdot (\beta_1 \hat{i} + \beta_2 \hat{j})$$

Using *distributive property*, we have

$$
\begin{align*}
\vec{v} \cdot \vec{w} &= 
\alpha_1 \beta_1 (\underbrace{\hat{i} \cdot \hat{i}}_1) + 
\alpha_1 \beta_2 (\underbrace{\hat{i} \cdot \hat{j}}_0) +
\alpha_2 \beta_1 (\underbrace{\hat{j} \cdot \hat{i}}_0) +
\alpha_2 \beta_2 (\underbrace{\hat{j} \cdot \hat{j}}_1) \\
&= 
\alpha_1 \beta_1 + \alpha_2 \beta_2 \\
\end{align*}
$$

which is the *definition* of **dot product** we have been familiar with.

###  Moving out of the unit basis vectors

Suppose we let the basis $$\hat{i}$$ and $$\hat{j}$$ be any vector in the 2D Cartesian plane, i.e. $$\hat{i} = \vec{b_1}$$ and $$\hat{j} = \vec{b_2}$$, the **dot product** of $$\vec{v}$$ and $$\vec{w}$$ cannot be simplified as above, 

$$
\begin{align*}
\vec{v} \cdot \vec{w} &= 
\alpha_1 \beta_1 (\hat{i} \cdot \hat{i}) + 
\alpha_1 \beta_2 (\hat{i} \cdot \hat{j}) +
\alpha_2 \beta_1 (\hat{j} \cdot \hat{i}) +
\alpha_2 \beta_2 (\hat{j} \cdot \hat{j}) \\
&=
\alpha_1 \beta_1 (\vec{b_1} \cdot \vec{b_1}) + 
\alpha_1 \beta_2 (\vec{b_1} \cdot \vec{b_2}) +
\alpha_2 \beta_1 (\vec{b_2} \cdot \vec{b_1}) +
\alpha_2 \beta_2 (\vec{b_2} \cdot \vec{b_2}) 
\end{align*}
$$

where $$(\vec{b_1} \cdot \vec{b_1}), (\vec{b_1} \cdot \vec{b_2}), (\vec{b_2} \cdot \vec{b_2}) $$ requires the specification of the two vectors $$\vec{b_1}$$ and $$\vec{b_2}$$.

In fact, we can rewrite the above in terms of **matrix mulitplication**,

$$\vec{v} \cdot \vec{w} = 

\begin{bmatrix} \alpha_1 & \alpha_2 \\ \end{bmatrix}

\begin{bmatrix}
\vec{b_1} \cdot \vec{b_1} & \vec{b_1} \cdot \vec{b_2} \\
\vec{b_2} \cdot \vec{b_1} & \vec{b_2} \cdot \vec{b_2}
\end{bmatrix}

\begin{bmatrix}
\beta_1 \\
\beta_2
\end{bmatrix}$$

We can denote it simply as,

$$\alpha^T M \beta = \beta^T M \alpha$$

where $$M$$ is the **Gram Matrix**.

### Gram Matrix

In general, the **Gram Matrix** can take in multiple vectors $$b_1, \dots, b_n$$

$$M = 
\begin{bmatrix}
\vec{b_1} \cdot \vec{b_1} & \cdots &  \vec{b_1} \cdot \vec{b_n}\\
\vdots & \ddots & \vdots \\
\vec{b_n} \cdot \vec{b_1} & \cdots & \vec{b_n} \cdot \vec{b_n}
\end{bmatrix}$$

where element in row $$j$$ and column $$k$$ is

$$M_{ij} = \vec{b_i} \cdot \vec{b_j}$$

Notice the diagonal line always have **dot products** of the particular vector of itself.
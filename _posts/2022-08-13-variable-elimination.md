---
layout: post
title:  Variable Elimination
date: 2022-08-13 22:00:00 +0800
permalink: /post/ve/

working: true
hide: true
---

# Variable Elimination

This is currently a study note for [this](https://ermongroup.github.io/cs228-notes/inference/ve/){:target="_blank"}

### Variable Elimination in a simple Bayesian Network

Assume $$x_i$$ are *discrete variables* and each are assigned *k* values.

Given a *joint probability* $$p(x_1, \dotsc, x_n)$$, we can find $$p(x_n)$$ through *marginalizing* $$x_1, x_2, ..., x_{n-1}$$,

$$p(x_n) = \sum_{x_1} \cdots \sum_{x_{n-1}} p(x_1, \dotsc, x_n)$$

Since we consider a simple *Bayesian Network*, where $$x_i$$ only depends on $$x_{i-1}$$ $$(x_i \perp \{x_1...x_{i-2}\})$$, then 

$$p(x_1, \dotsc, x_n) = p(x_1) \prod_{i=2}^n p(x_i \mid x_{i-1})$$

Putting the above two together , we can get,

$$
\begin{align*}
p(x_n) &= \sum_{x_1} \cdots \sum_{x_{n-1}} p(x_1, \dotsc, x_n) \\
& = \sum_{x_1} \cdots \sum_{x_{n-1}} p(x_1) \prod_{i=2}^n p(x_i \mid x_{i-1}) \\

& = \sum_{x_{n-1}} p(x_n \mid x_{n-1}) \sum_{x_{n-2}} p(x_{n-1} \mid x_{n-2}) \cdots \sum_{x_2} p(x_3 \mid x_2) {\color{dodgerblue} {\underbrace{\sum_{x_1} p(x_2 \mid x_1) p(x_1)}_{\text{marginal } p(x_2)}}} \\

& = \sum_{x_{n-1}} p(x_n \mid x_{n-1}) \underbrace{\sum_{x_{n-2}} p(x_{n-1} \mid x_{n-2}) \cdots \underbrace{\sum_{x_2} p(x_3 \mid x_2) {\color{dodgerblue} {p(x_2)}}}_{\text{marginal }p(x_3)}}_{\text{marginal }p(x_{n-1})} \\

& = \underbrace{\sum_{x_{n-1}} p(x_n \mid x_{n-1}) p(x_{n-1})}_{\text{marginal } p(x_n)} \\
\end{align*}
$$

Calculating the $${\color{dodgerblue} {\text{marginal } p(x_2)}}$$ eliminates variable $$x_1$$. The process goes on and on until we get the $$\text{marginal } p(x_n)$$. This is like eliminating variable $$x_1, x_2, ..., x_{n-1}$$ one by one. We name this process **variable elimination**. 

### Time Complexity

Remember each discrete variable $$x_i$$ are assigned $$k$$ values, $$x_1, x_2, ..., x_{n-1}$$ takes on a total $$k^{n-1}$$ values. 

In the above process, the first *marginalizing* step for $${\color{dodgerblue} {\text{marginal } p(x_2)}}$$ is considering each $$x_1$$ for each $$x_2$$, so it takes $$O(k^2)$$ time.

The remaining *marginalizing* steps are doing the same process, so in total there are $$n$$ *marginalizing* steps. The whole *variable elimination* for the above *Bayesian network* takes $$O(nk^2)$$ time.

### Variable Elimination in a Bayesian Network

Consider the following student *Bayesian Network* example, 

![student_bn](/assets/img/student-bn.png "student BN example")

### Variable Elimination in a general form

In the above example, we assume a *Bayes Network* which is a special case of such elimination. We now consider a general case of such process.
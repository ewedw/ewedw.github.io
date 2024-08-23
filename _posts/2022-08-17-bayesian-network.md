---
layout: post
title:  Bayesian Network
date: 2022-08-17 22:00:00 +0800
permalink: /post/bn/

hide: true
---

# Bayesian Network

This is a concise note capturing some essence in Bayesian Network. Please read the book [PGM](https://github.com/kushagra06/CS228_PGM/blob/master/Probabilistic%20Graphical%20Models%20-%20Principles%20and%20Techniques.pdf){:target="_blank"} (a really great book) for in-depth discussions of technical details.

### From the Chain Rule to Directed Acyclic Graph (DAG)

Remember the *product rule*, 

$$P(X_1, X_2) = P(X_1) P(X_2 \mid X_1)$$

then generalizing it to the *chain rule*,

$$
\begin{align*}
&\ P(X_1, X_2, \dotsc ,X_n) \\
& = P(X_1)P(X_2 \mid X_1)P(X_3 \mid X_1, X_2) \dotsb P(X_n \mid X_1, X_2, \dotsc ,X_{n-1})\\
& = P(X_1) \prod_{i=2}^n P(X_i \mid X_{< i})
\end{align*}
$$

The *chain rule* enables us to use factors of *conditional probability* to represent the *joint probability* $$P(X_1, X_2, \dotsc ,X_n)$$.

In fact, we can also use a graph to represent it. A **directed acyclic graph (DAG)**

* contains **vertices**

* contains **edges** directed from one vertex to another

* never forms a **closed loop** 

Here, we represent the *joint probability* $$P(X_1, X_2, \dotsc ,X_n)$$ with *vertices* as $$\require{enclose}\Large{\enclose{circle}{X_i}}$$ and *edges* as $$\longrightarrow$$.

![fcbn](\assets\img\fcbn.png "Fully connected Bayesian Network")

We call this *DAG* a **fully connected Bayesian Network**.


### Time complexity

Suppose each variable $$X_i$$ takes on $$k$$ values, as $$n$$ becomes very large (which is often the case for model learning in practice), the time complexity grows exponentially. 

The last (most complex) factor $$P(X_n \mid X_1, X_2, \dotsc ,X_{n-1})$$ takes $$O(k^{n-1})$$ time. Even if the variable is boolean where $$k=2$$, the last term still takes $$O(2^{n-1})$$ time. 

In total, anayzing each factor, it takes

$$O(k^0+k^1+k^2+\dotsc+k^{n-1}) = O(k^n-1)$$

time.

Consider an image input with size $$256 \times 256$$. If each pixel corresponds to a variable $$X_i$$, there are a total of $$n = 256^2 = 65536$$ variables. Since $$X_i \in [0,255]$$ takes on $$k=256$$ values, the term $$P(X_n \mid X_1, X_2, \dotsc ,X_{n-1})$$ will take 

$$O(k^{n-1}) = O(256^{65536-1}) = O(256^{65535})$$ 

time. That's *one* single image only.

### Introducing Independence

One way to reduce the complexity is to introduce **independence**.

Assume the variable $$X_i$$ is dependent only on a few variables before and we denote the set of dependent variables as $${\color{dodgerblue} X_i^{(Dep)}}$$, then the *conditional probability*

$$P(X_i \mid X_1, \dotsc ,X_{i-1}) = P(X_i \mid {\color{dodgerblue} X_i^{(Dep)}})$$

For example, if $$X_i$$ is dependent only on $$X_{i-1}, X_{i-3}, X_1$$, then 

$$
\begin{gather}
{\color{dodgerblue} X_i^{(Dep)} = \{X_{i-1}, X_{i-3}, X_1\}} \\
P(X_i \mid X_1, \dotsc ,X_{i-1}) = P(X_i \mid {\color{dodgerblue} X_{i-1}, X_{i-3}, X_1})
\end{gather}
$$

Using *DAG*,  

![idep](\assets\img\indep.png "Independent in Bayesian Network")

We can also express the condition using *independence*, 

$$\{X_i \perp X_{i-2}, X_{2 \dotsc i-4}\}$$

### Markov Chain

If we consider $$X_i$$ depends only on the previous variable $$X_{i-1}$$, then we have a **Markov Chain**, 

$$P(X_1, \dotsc ,X_n) = \prod_{i=2}^n P(X_i \mid X_{i-1})$$

Using *DAG*,

![markov-chain](\assets\img\markov-chain.png "Markov Chain")

Using *independence*, 

$$\{X_i \perp X_{1 \dotsc i-2}\}$$

### Naive Bayes

In a simplified (naive) case, $$X_i$$ can be independent to all other variables except for $$X_1$$, then we have a **Naive Bayes Model**,

$$
\begin{align*}
P(X_1, \dotsc ,X_n) & = P(X_1)P(X_2 \mid X_1)P(X_3 \mid X_1) \dotsb P(X_n \mid X_1) \\
& = P(X_1) \prod_{i=2}^n P(X_i \mid X_1)
\end{align*}
$$

Using *DAG*, 

![naive-bayes](\assets\img\naive-bayes.png "Naive Bayes")

Using *independence*,

$$\{X_i \perp X_{2 \dotsc i-1}\}$$

### Total independence

In extreme, $$X_i$$ can be independent to all other variables, then we have,

$$\begin{align*}
P(X_1, \dotsc ,X_n) &= P(X_1)P(X_2) \dotsb P(X_n) \\
&= \prod_{i=1}^n P(X_i)
\end{align*}
$$

Using graph (it is NOT a *DAG* since there is no directed edge),

![total-indep](\assets\img\total-indep.png 'Total independence')

Using *independence*,

$$\{X_i \perp X_{< i}\}$$

In practice, this will almost never be considered. Since a variable is often related to some other variables and we are interested in figuring out those relationship.

### Student Example

Consider the following example, we have 5 variables $$D, I, G, S, L$$ (all taken from the first letter). Each variables are assigned some values shown in value tables.

Using *DAG* (with value tables), 

![student_bn](\assets\img\student-bn.png "student BN example")

We can write the *joint probability* as, 

$$P(D,I,G,S,L) = P(I)P(S \mid I)P(D)P(G \mid I,D)P(L \mid G)$$

and the *independence* as,

$$
\begin{gather}
\{D \perp I\} \\
\{G \perp S \mid I\} \\
\{L \perp D, I, S \mid G\} \\
\end{gather}
$$
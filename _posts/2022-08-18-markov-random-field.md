---
layout: post
title:  Markov Random Field
date: 2022-08-18 19:00:00 +0800
permalink: /post/mrf/

working: true
hide: true
---

# Markov Random Field

This is a concise note capturing some essence in Markov Random Field. Please read the book [PGM](https://github.com/kushagra06/CS228_PGM/blob/master/Probabilistic%20Graphical%20Models%20-%20Principles%20and%20Techniques.pdf){:target="_blank"} (a really great book) for in-depth discussions of technical details.

### Undirected Graph

[*Bayesian Network*](\post\bn){:target="_blank"} is a *directed acyclic graph (DAG)*, while the **Markov Random Field** is an **undirected graph** that

* contains **vertices**

* contains **edges** which are **bidirectional** (or not directional)

For example, if we have four person $$A,B,C,D$$ and there are four pairs of relationship $$(A,B),(B,C),(C,D),(D,A)$$, we have an *undirected graph*

![abcd](\assets\img\undirected-abcd.png "Undirected graph: ABCD")

Note that since the relationship are *birdirectional*, 

$$(D,A) = (A,D)$$

There is no relationship for $$(A,C), (B,D)$$, so we can easily describe the *independence*, 

$$
\begin{gather}
\{A \perp C \mid B,D\} \\
\{B \perp D \mid A,C\}
\end{gather}
$$

### Unnormalized function 

Unlike [*Bayesian Network*](\post\bn){:target="_blank"}, we cannot use *chain rule* to split the *joint probability* $$P(A,B,C,D)$$ into factors of *conditional probability*. We need to do something else.

First, let's quantify the relationship $$(X,Y)$$, denote as $$\phi (X,Y)$$. We call this $$\phi (X,Y)$$ a *factor*. For example, if we want to model how likely two person $$X,Y$$ will meet in the coming week,

$$
\begin{align*}
\phi(X,Y) =
\begin{cases}
50 & \text{if } (X,Y) = \text{"couples"} \\
10  & \text{if } (X,Y) = \text{"friends"} \\
1  & \text{otherwise}.
\end{cases}
\end{align*}
$$

A possible case for the four person $$A,B,C,D$$ can be

![abcd-phi](\assets\img\undirected-abcd-phi.png "ABCD with phi")

By multiplying all the factors, we get 

$$
\begin{align*}
\hat{P}(A,B,C,D) &= \phi(A,B) \phi(B,C) \phi(C,D) \phi(D,A) \\
&= (1)(10)(50)(10) \\
&= 5000
\end{align*}
$$

We call $$\hat{P}$$ an **unnormalized function** instead of a probability distribution as the result may not lie within $$[0,1]$$.

### Partition function

To turn $$\hat{P}$$ into a true probability distribution $$P$$, we need to make *all  possible outcomes* sum to $$1$$. We can *normalized* the function $$\hat{P}$$ through dividing it by a **partition function** (here it is a constant), denote as $$Z$$

$$Z = \sum_{\phi \in \{A,B,C,D\}} \hat{P}(A,B,C,D)$$


So we can get the probability distribution $$P$$ from the *unnormalized function* $$\hat{P}$$,

$$P(A,B,C,D) = \frac{1}{Z} \hat{P}(A,B,C,D)$$

You can think of the *partition function* $$Z$$ as finding the total value of all $$\hat{P}$$. In this four person case, there are $$4$$ pairs of relationship $$(A,B),(B,C),(C,D),(D,A)$$ and each with $$3$$ possible values $$\{1, 10, 50\}$$, there are a total of $$3^4 = 81$$ outcomes for $$\hat{P}$$. The *partition function* $$Z$$ then have $$81$$ outcomes to add,

$$Z = \underbrace{(1)(1)(1)(1) + \dotsc + (50)(50)(50)(50)}_{81 \text{ outcomes}}$$

Using python to calculate $$Z$$, 

```
values = [1, 10, 50]
num_of_outcome = 0
total = 0

for a in values:
    for b in values:
        for c in values:
            for d in values:
                print(a, b, c, d) # print all outcomes
                num_of_outcome += 1
                total += a*b*c*d
                
print(num_of_outcome) # should be 81
print(total) # should be 13845841 
```

We have $$Z = 13845841$$, so the probability for the above case

$$
\begin{align*}
P(A,B,C,D) &= \frac{1}{13845841} \hat{P}(A,B,C,D) \\
&= \frac{1}{13845841} (5000) \\
&\approx 0.00036 
\end{align*}
$$



### Formal definition

The probability distribution $$P$$ for an *undirected graph* of vertices $$X_1, \dotsc, X_n$$, 

$$P(X_1, \dotsc, X_n) = \frac{1}{Z} \prod_{c \in C} \phi_c(X_c)$$

where $$C$$ denotes a set of *cliques* in the graph and $$Z$$ denotes the *partition function* for normalization,

$$Z = \sum_{X_1, \dotsc, X_n} \prod_{c \in C} \phi_c(X_c)$$

---
layout: post
title:  Probability Review
date: 2022-08-12 17:00:00 +0800
permalink: /post/probability/

working: true
hide: true
---

<!-- <span style="color:#FF8970;"></span> -->

# Probability Review

Here is a **review** for probability theory, an essential foundation for understanding probabilistic graphical models. I try to include some graphs for easier interpretation. 

### Events

* **Sample space** $$\Omega$$: The set of **all** possible outcomes.  
&emsp;i.e. $$\Omega = \{\text{Pen}_{\color{dodgerblue} blue}, \text{Pen}_{\color{red} red}, \text{Pen}_{\color{forestgreen} green}, \text{Pen}_{black}\}$$ in my bag

$$
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\
                   &       &       &       &             \\ 
                   &       &\Omega &       &             \\ 
                   &       &       &       &             \\  
                   &       &       &       &             \\ 
\hline
\end{array}
$$

* **Event** $$E$$: A subset of the sample space $$E \subseteq \Omega$$.  
&emsp;i.e. $$E_1 = \{\text{Pen}_{\color{dodgerblue} blue}\}$$  
&emsp;i.e. $$E_2 = \{\text{Pen}_{\color{forestgreen} green}\}$$  
&emsp;i.e. $$E_3 = \{\text{Pen}_{black}, \text{Pen}_{\color{red} red}\}$$  
&emsp;i.e. $$E_4 = \{\} = \varnothing$$ (an *empty set*)

$$
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\
             E_1   &  E_3  &       &       &             \\ 
                   &       &\Omega &       &             \\ 
                   &       &       &       &     E_2     \\  
                   &       &       &       &             \\ 
\hline
\end{array}
$$

* **Event space** $$\mathcal{E}$$: All events $$E_1, E_2, E_3, E_4, ... \in \mathcal{E}$$.

### Axioms of Probability

**Probability measures $$P$$** is a function $$P : \mathcal{E} \to \mathbb{R}$$ that has the following properties,

1. $$P(E) \ge 0$$ (for all $$E \in \mathcal{E}$$).

2. $${\color{dodgerblue} P(\Omega) = 1}$$.

3. If $$E_1, E_2, E_3,...$$ are *disjoint* events, then

$$
\begin{align*}
P(E_1 \cup E_2 \cup E_3 \cup ...) &= P(E_1) + P(E_2) + P(E_3) + ... \\
P(\bigcup_{n=1}^{\infty} E_n) &= \sum_{n=1}^{\infty} P(E_n)
\end{align*}
$$

### Properties

* If an event $$E$$ is **impossible**, it is the *empty set* $$\varnothing$$,

$$P(E) = P(\varnothing) = 0$$

* If an event $$E$$ is **surely possible**, it is the *sample space*
$$\Omega$$, 

$$P(E) = P(\Omega) = 1$$

$$
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\
                   &       &       &       &             \\ 
                   &       &\Omega &       &             \\ 
                   &       &       &       &             \\  
                   &       &       &       &             \\ 
\hline
\end{array}
$$

* If $$E_1$$ and $$E_2$$ are **disjoint**, $$E_1 \cap E_2$$ is an *empty set* $$\varnothing$$, 

$$\underbrace{P(E_1 \cap E_2) = P(\varnothing) = 0}_{\text{no overlap }\cap}$$

$$
\begin{array}{|c|c|c|c|c|}
\hline \quad & E_1 & \quad & E_2 & \quad\,\, \\
             & E_1 &       & E_2 &           \\ 
             & E_1 &       & E_2 &           \\ 
             & E_1 &       & E_2 &           \\  
             & E_1 &       & E_2 &           \\ 
\hline
\end{array}
$$

* If $$E_1$$ and $$E_2$$ are **joint**,  

$$P(E_1 \cup E_2) = P(E_1) + P(E_2) {\color{dodgerblue} - \underbrace{P(E_1 \cap E_2)}_{\text{extra overlap }\cap}}$$

$$
\require{cancel} 
\begin{array}{|cc|c|cc|}
\hline     &     &     E_1    &     &     \\
           &     &     E_1    &     &     \\ 
\hline E_2 & E_2 &  \cap \xcancel{\cap} & E_2 & E_2 \\ 
\hline     &     &     E_1    &     &     \\  
           &     &     E_1    &     &     \\ 
\hline
\end{array}
$$

* If $$E$$ and $$E'$$ are **complementary**,  they add up to the *sample space* $$\Omega$$,

$$
\begin{align*}
E + E' &= \Omega \\
P(E) + P(E') &= P(\Omega) \\
&= 1 \\
P(E') &= 1 - P(E) \\
\end{align*}
$$

$$
\begin{array}{|ccc|c|}
\hline \quad & \quad & \quad & \qquad \qquad \\
             &       &       &               \\ 
             & E     &       &       E'      \\ 
             &       &       &               \\  
             &       &       &               \\ 
\hline
\end{array}
$$


### Product Rule

Assume $$E_1, E_2$$ be 2 *possible* events, i.e. $$P(E_{1,2}) > 0$$. We denote the *joint* event as $$E_1 \cap E_2$$ and the **joint probability** as $$P(E_1 \cap E_2)$$. We can calculate it using the **product rule**,

$$P(E_1 \cap E_2) = P(E_1){\color{dodgerblue} P(E_2 \mid E_1)}$$

$$
\begin{array}{|cc|c|cc|}
\hline     &     &     E_1    &     &     \\
           &     &     E_1    &     &     \\ 
\hline E_2 & E_2 &  \cap \cap & E_2 & E_2 \\ 
\hline     &     &     E_1    &     &     \\  
           &     &     E_1    &     &     \\ 
\hline
\end{array}
\begin{array}{ccc}
&&\\ 
& P(E_1) &\\
& \longrightarrow &\\ 
&&\\ 
&& \\
\end{array}
\begin{array}{|c|}
\hline  E_1 \\
        E_1 \\ 
\hline \cap \\ 
\hline  E_1 \\  
        E_1 \\ 
\hline
\end{array}
\begin{array}{ccc}
&&\\ 
& {\color{dodgerblue} P(E_2 \mid E_1)} &\\
& \longrightarrow &\\ 
&&\\ 
&& \\
\end{array}
\begin{array}{|c|} 
\hline \cap \\ \hline 
\end{array}
$$

### Conditional Probability

From above, $${\color{dodgerblue} P(E_2\mid E_1)}$$ is the **conditional probability** of $$E_2$$ given observing the outcome $$E_1$$. It is usually presented as,

$${\color{dodgerblue} P(E_2 \mid E_1)} = \frac{P(E_1 \cap E_2)}{P(E_1)}$$


### Independence

Asssume $$E_1, E_2$$ are **independent**, then observing the outcome $$E_1$$ has no effect on the probability of $$E_2$$. It means $$P(E_2 \mid E_1) = P(E_2)$$. Using *product rule*,

$$
\begin{align*}
&\ P(E_1 \cap E_2) \\
& = P(E_1){\color{dodgerblue}P(E_2 \mid E_1)}\\
& = P(E_1){\color{dodgerblue}P(E_2)}
\end{align*}
$$


$$\text{Area }E_1 = 1/5$$

$$
\begin{array}{|cc|c|cc|}
\hline \quad\,\,\, & \quad &  E_1  & \quad & \quad\,\,\, \\
                   &       &  E_1  &       &             \\ 
                   &       & \cap  &       &             \\ 
                   &       &  E_1  &       &             \\  
                   &       &  E_!  &       &             \\ 
\hline
\end{array}
$$

$$\text{Area }E_2 \mid E_1 = 1/5$$

$$
\begin{array}{|c|}
\hline  E_1 \\
        E_1 \\ 
\hline \cap \\ 
\hline  E_1 \\  
        E_1 \\ 
\hline
\end{array}
$$

$$\text{Area }E_2 = 1/5$$

$$
\begin{array}{|ccccc|}
\hline     &     &        &     &     \\
           &     &        &     &     \\ 
\hline E_2 & E_2 &  \cap  & E_2 & E_2 \\ 
\hline     &     &        &     &     \\  
           &     &        &     &     \\ 
\hline
\end{array}
$$


Note that "*independent*" is different from "*disjoint*",

* "*Disjoint*" events **cannot** occur at the same time.    
&emsp;i.e. $$E_1 = \{ \text{you being at home} \}$$  
&emsp;i.e. $$E_2 = \{ \text{you being in the park} \}$$  
&emsp;i.e. $$P(E_1 \cap E_2) = 0$$

* "*Independent*" events **can** occur at the same time, so they are *joint*.   
But observing one outcome has no effect on other outcomes.  
&emsp;i.e. $$E_1 = \{ \text{you being at home} \}$$  
&emsp;i.e. $$E_3 = \{ \text{me being at home} \}$$  
&emsp;i.e. $$P(E_1 \cap E_3) > 0$$ 

* "*Dependent*" events,
&emsp;i.e. $$E_1 = \{ \text{you being at home} \}$$  
&emsp;i.e. $$E_2 = \{ \text{you being in the park} \}$$  
&emsp;i.e. $$E_4 = \{ \text{you are sitting} \}$$
&emsp;i.e. If you are at home, you are mostly sitting.
&emsp;i.e. If you are in the park, you rarely sit.
&emsp;i.e. $$P(E_4 \mid E_1) \neq P(E_4 | E_2)$$, it depends ~

### Bayes' theorem

In fact, the *joint* event $$E_1 \cap E_2$$ is the same as $$E_2 \cap E_1$$. Using *product rule*,

$$
\begin{align*}
P(E_1 \cap E_2) &= P(E_2 \cap E_1)\\
P(E_1)P(E_2 \mid E_1) &= P(E_2)P(E_1 \mid E_2)\\
\end{align*}
$$

It simply means we can deduce *conditional probability* in a different order.

If we divide both side by $$P(E_1)$$, we get the **Bayes' theorem**,

$$P(E_2 \mid E_1) = \frac{P(E_2)P(E_1 \mid E_2)}{P(E_1)}$$

Personally, I simply keep in mind $$P(E_1 \cap E_2) = P(E_2 \cap E_1)$$.


### Chain Rule

What if there are more than 2 possible events?

Assume $$E_1, E_2, E_3, E_4$$ be 4 possibe events, i.e. $$P(E_{1,2,3,4}) > 0$$, using *product rule* continuously,

$$
\begin{align*}
&\ P(E_1 \cap E_2 \cap E_3 \cap E_4) \\
& = P({\color{dodgerblue} E_1 \cap E_2 \cap E_3})P(E_4 \mid {\color{dodgerblue} E_1 \cap E_2 \cap E_3}) \\
& = P({\color{dodgerblue} E_1 \cap E_2})P(E_3 \mid {\color{dodgerblue} E_1 \cap E_2})P(E_4 \mid E_1 \cap E_2 \cap E_3) \\
& = P({\color{dodgerblue} E_1})P(E_2 \mid {\color{dodgerblue} E_1})P(E_3 \mid E_1 \cap E_2)P(E_4 \mid E_1 \cap E_2 \cap E_3)
\end{align*}
$$

Now, we can generalize it to $$k$$ possible events...

Assume $$E_1, E_2, ..., E_k$$ be $$k$$ possible events, i.e. $$P(E_{1,2,...,k}) > 0$$, it follows that,

$$
\begin{align*}
&\ P(E_1 \cap E_2 \cap \dotsb \cap E_k) \\
& = P(E_1)P(E_2 \mid E_1)P(E_3 \mid E_1 \cap E_2) \dotsb P(E_k \mid E_1 \cap E_2 \cap \dotsb \cap E_{k-1})
\end{align*}
$$

Remember that it can be deduced in a different order (i.e. in reverse),

$$
\begin{align*}
&\ P(E_1 \cap E_2 \cap \dotsb \cap E_k) \\
& = P(E_k \cap E_{k-1} \cap \dotsb \cap E_1) \\
& = P(E_k)P(E_{k-1} \mid E_k)P(E_{k-2} \mid E_k \cap E_{k-1}) \dotsb P(E_1 \mid E_k \cap E_{k-1} \cap \dotsb \cap \ E_2)
\end{align*}
$$

The **Chain rule** can simply be written as, 

$$P(\bigcap_{n=1}^k E_n) = P(E_1) \prod_{n=2}^k P(E_n \mid \bigcap E_{< n})$$

<hr>

### Random variable

Till now, our inputs to the probability measure $$P$$ are some events $$E$$. 

* *Events* $$E$$ are not necessary measurable.  
&emsp;i.e. $$E_1 = \{\text{Pen}_{\color{dodgerblue} blue}\}$$ in my bag  
&emsp;i.e. $$E_h = \{\text{Head}\}$$ in tossing a coin


A **random variable $$X$$** is a function $$X : \Omega \to \mathcal{M}$$, where $$\mathcal{M}$$ is some *measurable space*.  

$$
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\
                   &       &       &       &             \\ 
                   &       &\Omega &       &             \\ 
                   &       &       &       &             \\  
                   &       &       &       &             \\ 
\hline
\end{array}
\begin{array}{ccc}
&&\\ 
& X &\\
& \longrightarrow &\\  
&&\\ 
&&\\
\end{array}
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\
                   &       &       &       &             \\ 
                   &       &\mathcal{M} &       &             \\ 
                   &       &       &       &             \\  
                   &       &       &       &             \\ 
\hline
\end{array}
$$


* *Random variables* $$X$$ are measurable.  
&emsp;i.e. $$X_1 = \text{Number of Pen}_{\color{dodgerblue} blue}$$ in my bag  
&emsp;i.e. $$X_h = \text{Number of Head}$$ tossed in 50 times

* We can get a *random variable* $$X(E)$$ from an event $$E$$.  
&emsp;i.e. *Sample space* $$\Omega = \{\text{brightness of a pixel}\}$$  
&emsp;i.e. *Measurable space* $$\mathcal{M} = \{0, 1, 2, ..., 255\}$$ for all pixel values  
&emsp;i.e. *Event* $$E_w = \{\text{total white}\}$$  
&emsp;i.e. *Random Variable* $$X(E_w) = 255$$

$$
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\
                   &       &       &       &             \\ 
                   &       &\Omega &       &             \\ 
                   &       &       &       &             \\  
                   &       &       &  E_w  &     \to    \\ 
\hline
\end{array}
\begin{array}{ccc}
&&\\ 
&&\\
&&\\ 
& X(E_w) &\\ 
& \longrightarrow & \\
\end{array}
\begin{array}{|ccccc|}
\hline \quad\,\,\, & \quad & \quad & \quad & \quad \\
                   &       &       &       &             \\ 
                   &       &\mathcal{M} &       &             \\ 
                   &       &       &       &             \\  
               \to    &   \to    &   \to    &   255   &             \\ 
\hline
\end{array}
$$

* We can also assign a value $$x$$ to the *random variable* $$X$$, denote as $$X = x$$.  
&emsp;i.e. Assigning *random variable* $$X = 160$$  
&emsp;i.e. Implying *event* $$E_g = \{\text{grey}\}$$

$$
\begin{array}{|ccccc|}
\hline             &  E_g  & \gets & \gets &   \gets     \\
                   &       &       &       &             \\ 
                   &       &\Omega &       &             \\ 
                   &       &       &       &             \\  
       \quad\,\,\, & \quad & \quad & \quad & \quad\,\,\, \\ 
\hline
\end{array}
\begin{array}{ccc}
& \longleftarrow &\\ 
& \text{implies} &\\
&&\\ 
&&\\ 
&& \\
\end{array}
\begin{array}{|ccccc|}
\hline       \gets & \gets & \gets &   160  &             \\  
                   &       &       &        &             \\ 
                   &       &\mathcal{M}     &&            \\ 
                   &       &       &        &             \\  
       \quad\,\,\, & \quad & \quad & \quad  & \quad       \\
\hline
\end{array}
$$

With *random variable* $$X$$ as input, we now can determine a computable function $$F$$ for the probability measure $$P$$. 


### Cumulative Distribution Function

### Probability Mass Function (Discrete)

### Probability Density Function (Continuous)

### Expectated value

### Variance

### Common Distributions

<hr>

### Joint Distribution

### Marginalization

### Extending the Bayes theorems

### Expectation

### Covariance

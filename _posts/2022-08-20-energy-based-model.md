---
layout: post
title:  Energy-based Model
date: 2022-08-20 23:00:00 +0800
permalink: /post/energymodel/

working: true
hide: true
---

# Energy-based Model

Remember two important points for a probability distribution $$P(x)$$

* Non-negative

$$P(x) \ge 1$$

* Sum to 1

$$
\begin{align*}
\sum_x P(x) &= 1  \quad\text{(for discrete variables)}\\
\int P(x)dx &= 1 \quad\text{(for continuous variables)}
\end{align*}
$$

We can actually make any function $$f(x)$$ sum to 1, by

$$P(x) = \frac{1}{Z} f(x)$$

where $$Z$$ is the *normalizing constant* called the **partition function**,

$$
\begin{align*}
Z =
\begin{cases}
\displaystyle\sum_x f(x) & \text{(for discrete variables)} \\
\displaystyle\int f(x)dx  & \text{(for continuous variables)}
\end{cases}
\end{align*}
$$

### Energy-based model training

Here we introduce $$\theta$$ as the weight that can be learnt during training. We denote the function $$f(x)$$ with weight $$\theta$$ as $$f_\theta(x)$$. Since we are interested in the *log-likehood* of $$P_\theta(x)$$ in later time, it is nice to exponentiate $$f_\theta(x)$$,

$$P_\theta(x) = \frac{1}{Z(\theta)} \, e^{\,f_\theta(x)}$$

where

$$
\begin{align*}
Z(\theta) =
\begin{cases}
\displaystyle\sum_x e^{\,f_\theta(x)} & \text{(for discrete variables)} \\
\displaystyle\int e^{\,f_\theta(x)}dx  & \text{(for continuous variables)}
\end{cases}
\end{align*}
$$

This **energy-based** form is actually highly relevant (i.e. thermodynamics). When we use $$-f_\theta(x)$$ instead of $$f_\theta(x)$$ as the power, the term $$-f_\theta(x)$$ is actually called the **energy**. 

Many common distributions can actually be expressed in this form (you can try it out). For example, *Gaussian distribution* takes the *energy* $$-f_\theta(x)$$ as

$$-f_\theta(x) = -f_{\mu,\sigma}(x) = -\frac{(x-\mu)^2}{2\sigma^2}$$

For details, please see [**Gaussian Distribution PDF**](\post\gaussian){:target="_blank"}.

Pros:

* Flexible: $$f_\theta(x)$$ can be any function

Cons:

* Curse of dimensionality: Calculating $$P_\theta(x)$$ is hard since $$Z(\theta)$$ is *intractable*

* Sampling is hard: Since $$Z(\theta)$$ is *intractable*

* Learning is hard: **Maximum Likelihood Estimation (MLE)** is hard

If we can find a way to work around with $$Z(\theta)$$, maybe through sampling or simply bypassing it, this **energy-based model** can be very useful.

### Evaluating ratios between two variables

If we have two variables $$x_1$$ and $$x_2$$, we can evaluate their ratio $$P_\theta(x_1)$$ and $$P_\theta(x_2)$$ without the need to handle $$Z(\theta)$$

$$
\frac{P_\theta(x_1)}{P_\theta(x_2)} 
= \cfrac{\cfrac{1}{Z(\theta)} \, e^{\,f_\theta(x_1)}}{ \cfrac{1}{Z(\theta)} \, e^{\,f_\theta(x_2)}} 
= \frac{e^{\,f_\theta(x_1)}}{e^{\,f_\theta(x_2)}}
= e^{\,f_\theta(x_1) - f_\theta(x_2)}
$$

This is very useful in MCMC. As we will be calculating the **acceptance probability** (explained below)

### Contrastive Divergence

In many cases, we want to learn by **Maximum Likelihood Estimation (MLE)**. Here we do the same. To maximize the *log-likelihood* of $$P_\theta(x)$$, we need to compute its *gradient*. Let's simplify the *log-likelihood* first, 

$$
\begin{align*}
\ln P_\theta(x) &= \ln \left( \frac{1}{Z(\theta)} \, e^{\,f_\theta(x)} \right) \\
&= \ln \left( e^{\,f_\theta(x)} \right) - \ln Z(\theta) \\
&= f_\theta(x) - \ln Z(\theta)
\end{align*}
$$

Taking the *gradient* of the *log-likelihood* with respect to $$\theta$$,

$$
\begin{align*}
\nabla_\theta \ln P_\theta(x) 
&= \nabla_\theta\, f_\theta(x) - \nabla_\theta \, \ln Z(\theta) \\
&= \nabla_\theta\, f_\theta(x) - \frac{\nabla_\theta \, Z(\theta)}{Z(\theta)} \\
&= \nabla_\theta\, f_\theta(x) - \frac{1}{Z(\theta)} \int \nabla_\theta \, e^{\,f_\theta(x)} dx \\
&= \nabla_\theta\, f_\theta(x) - \frac{1}{Z(\theta)} \int e^{\,f_\theta(x)} \, \nabla_\theta\, f_\theta(x) dx \\
&= \nabla_\theta\, f_\theta(x) - \int \frac{1}{Z(\theta)} \, e^{\,f_\theta(x)} \, \nabla_\theta\, f_\theta(x) dx \\
&= \nabla_\theta\, f_\theta(x) - \int P_\theta(x) \, \nabla_\theta\, f_\theta(x) dx \\
&= \nabla_\theta\, f_\theta(x) - \mathbb{E}_{P_\theta(x)} [\nabla_\theta\, f_\theta(x)]
\end{align*}
$$

Calculating $$\nabla_\theta\, f_\theta(x)$$ is relatively easy.

But calculating the *expected value* $$\mathbb{E}_{P_\theta(x)} [\nabla_\theta\, f_\theta(x)]$$ is almost impossible, as it contains the computation of $$P_\theta(x)$$ and thus the *intractable* $$Z(\theta)$$. So we may first consider estimating it with sampling,

$$
\nabla_\theta \ln P_\theta(x) \approx \nabla_\theta\, f_\theta(x_{train}) - \nabla_\theta\, f_\theta(x_{sample})
$$

We call this the **Contrastive Divergence**.

### Sampling with MCMC

Now we have a formula for estimating the *maximium log-likelihood*, but it requires us to do sampling. How do we do sampling? We can use a sampling method called **Markov Chain Monte Carlo (MCMC)** to approximate $$P_\theta(x)$$.

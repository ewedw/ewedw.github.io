---
layout: post
title:  Gaussian Distribution PDF
date: 2022-08-21 03:00:00 +0800
permalink: /post/gaussian/

---

# Gaussian Distribution PDF

Here we are deriving the general form of the **probability density function (PDF)**  for **Gaussian distribution** using [**energy-based model**](\post\ebm){:target="_blank"}.

The *energy-based model* takes the form of 

$$P_\theta(x) = \frac{1}{Z(\theta)} \, e^{\,-f_\theta(x)}$$

where $$\theta$$ is the given weight and $$Z(\theta)$$ is the **partition function** for *normalizing* $$\displaystyle e^{\,-f_\theta(x)}$$,

$$
\begin{align*}
Z(\theta) =
\begin{cases}
\displaystyle\sum_x e^{\,f_\theta(x)} & \text{(for discrete variables)} \\
\displaystyle\int e^{\,f_\theta(x)}dx  & \text{(for continuous variables)}
\end{cases}
\end{align*}
$$


### Here we go

In *Gaussian distribution*, the weight $$\theta$$ takes in $$2$$ values: the *mean* $$\mu$$ and the *standard deviation* $$\sigma$$. The *energy* $$-f_\theta(x)$$ takes

$$-f_\theta(x) = -f_{\mu,\sigma}(x) = -\frac{1}{2} \left( {\color{dodgerblue} \frac{x-\mu}{\sigma}} \right)^2 = -\frac{(x-\mu)^2}{2\sigma^2}$$

Notice $$\displaystyle {\color{dodgerblue} \frac{x-\mu}{\sigma}}$$ is the **standard score** of $$x$$, the *energy* $$-f_\theta(x)$$ actually describes the *deviation* of $$x$$ compared to the *mean* $$\mu$$.

With $$\theta = \mu, \sigma$$, we can write $$P_\theta(x)$$ as

$$P_{\mu,\sigma}(x) = \frac{1}{Z(\mu,\sigma)} \, e^{\,-\frac{(x-\mu)^2}{2\sigma^2}}$$

### The partition function $$Z(\mu,\sigma)$$

Calculating $$Z(\mu,\sigma)$$ requires *integration*. Luckily, there is a *closed-form expression* for it,

$$Z(\mu,\sigma) = \int e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx$$

Let $$\displaystyle u = \frac{x-\mu}{\sqrt{2}\sigma}$$,

$$
\begin{align*} 
\frac{du}{dx} &= \frac{1}{\sqrt{2}\sigma} \\
dx &= \sqrt{2}\sigma \,du
\end{align*}
$$

Substitute $$u$$,

$$Z(\mu,\sigma) = \int e^{-u^2} \sqrt{2}\sigma \,du$$

We multiply $$\displaystyle\frac{\sqrt{2\pi}}{\sqrt{2\pi}}$$ as a special treatment, 

$$
\begin{align*}
Z(\mu,\sigma) &= \frac{\sqrt{2\pi}}{\sqrt{2\pi}} \int e^{-u^2} \sqrt{2}\sigma \,du \\
&= \frac{\sqrt{\pi}\sigma}{\sqrt{2}} {\color{dodgerblue} \int \frac{2e^{-u^2}}{\sqrt{\pi}} du}
\end{align*}
$$

$${\color{dodgerblue} \displaystyle \int \frac{2e^{-u^2}}{\sqrt{\pi}} du}$$ is a special *integral* function called **Gaussian error function**. We denote it as $${\color{dodgerblue} \text{erf}(u)}$$,

$$Z(\mu,\sigma) = \frac{\sqrt{\pi}\sigma}{\sqrt{2}} \, {\color{dodgerblue} \text{erf}(u)} + C
$$

Since we are finding the area under the curve, we would like to consider the *definite integral*,

$$
\begin{align*}
Z(\mu,\sigma) &= \int_{-\infty}^{+\infty} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx \\
&= \left[ \frac{\sqrt{\pi}\sigma}{\sqrt{2}} \, {\color{dodgerblue} \text{erf}(u)} + C \right]_{-\infty}^{+\infty}
\end{align*}
$$

Consider the graph for $${\color{dodgerblue} \text{erf}(u)}$$,

![erfu](\assets\img\erfu.png "erf(u)")

$$
\begin{align*}
\lim_{u \to +\infty} {\color{dodgerblue} \text{erf}(u)} &= +1 \\
\lim_{u \to -\infty} {\color{dodgerblue} \text{erf}(u)} &= -1
\end{align*}
$$

So we have

$$
\begin{align*}
Z(\mu,\sigma) &= \left[ \frac{\sqrt{\pi}\sigma}{\sqrt{2}} \, {\color{dodgerblue} \text{erf}(u)} + C \right]_{-\infty}^{+\infty} \\
&= \frac{\sqrt{\pi}\sigma}{\sqrt{2}} \left( \lim_{u \to +\infty} {\color{dodgerblue} \text{erf}(u)} \,- \lim_{u \to -\infty} {\color{dodgerblue} \text{erf}(u)} \right) \\
&= \frac{\sqrt{\pi}\sigma}{\sqrt{2}} (1-(-1)) \\
&= \frac{\sqrt{\pi}\sigma}{\sqrt{2}} (2) \\
&= \sqrt{2\pi}\sigma
\end{align*}
$$

### Finally

Putting everything together, the general form of the *PDF* for **Gaussian distribution** $$P_{\mu, \sigma}(x)$$,

$$P_{\mu, \sigma}(x) = \frac{1}{\sqrt{2\pi}\sigma} \, e^{\,-\frac{(x-\mu)^2}{2\sigma^2}}$$
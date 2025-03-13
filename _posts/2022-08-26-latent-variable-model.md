---
layout: post
title:  Latent Variable Model with VAE
date: 2022-09-26 13:00:00 +0800
permalink: /post/latent/

working: true
---

# Latent Variable Model with VAE

There are TWO parts.

1. Statistical Concepts in building [**latent variable models**](#introducing-latent-variable-z).

2. Architectural point of view with [**variational autoencoder (VAE)**](#variational-auto-encoder-vae).

<hr>

### Wrapping the Statistical Concepts

* [**Latent Variable $$z$$**](#introducing-latent-variable-z)

* [**Gaussian Mixture Model**](#gaussian-mixture-model)

* [**Maximum Likelihood Estimation**](#maximum-likelihood-estimation)

* [**ELBO**](#the-evidence-lower-bound-elbo)

* [**Variational Inference**](#variational-inference)

* [**Stocahstic Learning with Variational Inference**](#learning-with-variational-inference-and-stochastic-gradient-descent)

* [**Monte Carlo Sampling**](#monte-carlo-sampling-for-the-elbo-and-the-gradient-nabla_theta-nabla_phi)

* [**Reparameterization**](#reparameterization)

<hr>

### Modelling variable $$x$$

Let's consider a variable $$x$$ which is an image of a face. 

We may assume that it comes from a probability distribution $$p(x)$$ and try to model the *joint probability* of every pixels $$p(x_i, \dotsc, x_n)$$ using **auto-regressive model**. We can directly evaluate *likelihood* of $$p(x)$$ using **chain rule**. This is 

* straight forward

* easy to train  

BUT 

* the generation is sequential which may not be ideal

* no features are learnt

Is there a way to model and learn features of variable $$x$$?

### Introducing Latent Variable $$z$$ 

As we know, certain things affect how a face may look in an image. For example, gender, age, ethnicity, azimuth, eye color, skin tone etc... We denote these things as $$z$$. We call $$z$$ as a **latent variable** since we assume they have an effect on the variable $$x$$ but they are **unobserved** during training. 

![faces](\assets\img\faces.png "Faces")

This can be very helpful. In general, many things can fall into categories but are unobserved, for example, emails can fall into different categories, videos can fall into different categories. Introducing **latent varaiable** allow us to leverage our prior knowledge when defining a model. 

### Latent Variable Models

Our goal is to figure out the *joint probability* of variable $$x$$ and the **latent variable** $$z$$

$$p(x, z)$$

Such that we can *marginalize* $$z$$ to find $$p(x)$$,

$$
\begin{align*}
p(x) &= \sum_z p(x, z) \\
&= \sum_z p(z)p(x \mid z) \\
\text{or } &= \int p(z)p(x \mid z)dz \quad \text{(for continuous case)}
\end{align*}
$$

with the *prior* probability $$p(z)$$ and the *likelihood* $$p(x \mid z)$$

Intuitively, you can think of

* the *prior* $$p(z)$$:  
as the probability for a "category" (i.e. blonde hair)

* the *likelihood* $$p(x \mid z)$$:  
as the likelihood of an image $$x$$ given the "category" $$z$$ (i.e. image of a face with blonde hair)

### Gaussian Mixture Model

One common **latent variable model** is the **Gaussian Mixture Model**. Let's assume the *likelihood* $$p(x \mid z)$$ to be *Gaussian distributions*, for example,

$$p(x \mid z=\text{blonde hair}) = \mathcal{N}(x \mid \mu_z, \sigma_z)$$

* If we assume $$z$$ is *discrete* and represents categories, $$z \in \{1,\dotsc, K\}$$, we can then view the *marginal* $$p(x)$$ as a **Gaussian mixture model**, 

$$
\begin{align*}
p(x) &= \sum_z p(z)p(x \mid z) \\
&= \sum_z p(z) \mathcal{N}(x \mid \mu_z, \sigma_z)
\end{align*}
$$

![GMM](\assets\img\GMM.png "Gaussian Mixture Model")

![GMM2](\assets\img\GMM2.png "Gaussian Mixture Model 2")

* If we assume $$z$$ is *continuous*, $$p(z) = \mathcal{N}(z \mid 0,1)$$, it will be an **infinite Gaussian mixture model**.

$$
\begin{align*}
p(x) &= \int p(z) \mathcal{N}(x \mid \mu_z, \sigma_z) dz \\
&= \int \mathcal{N}(z \mid 0,1) \mathcal{N}(x \mid \mu_z, \sigma_z) dz
\end{align*}
$$

We can model $$\mu_z$$ and $$\sigma_z$$ using *Neural Networks* with weight $$\theta$$, denote as $$\mu_\theta(z)$$ and $$\sigma_\theta(z)$$

$$p_\theta(x \mid z) = \mathcal{N}(x \mid \mu_\theta(z), \sigma_\theta(z))$$

So we have the *marginal likelihood* $$p_\theta(x)$$ which depends on weight $$\theta$$,

$$
\begin{align*}
p_\theta(x) &= \sum_z p(z) \mathcal{N}(x \mid \mu_\theta(z), \sigma_\theta(z)) \\
&= \sum_z p(z) p_\theta(x \mid z) \\
&= \sum_z p_\theta(x, z) \\
\text{or } &= \int p_\theta(x, z) dz \quad \text{(for continuous case)}
\end{align*}
$$

### Maximum Likelihood Estimation

Now, we have the *marginal likelihood* $$p_\theta(x)$$ as a model with weight $$\theta$$. We hope to find the best $$\theta$$ so that it can best estimate the true distribution $$p(x)$$. How do we quantify the difference between the two distributions $$p_\theta(x)$$ and $$p(x)$$? We can use [**KL-divergence**](https://youtu.be/ErfnhcEV1O8){:target="_blank"}.

$$D_{KL}(p(x) \parallel p_\theta(x)) = \mathbb{E}_{x \sim p(x)} \left[ \ln \frac{p(x)}{p_\theta(x)} \right] \ge 0$$

We want to minimize the difference,

$$
\begin{align*}
\min_\theta D_{KL}(p(x) \parallel p_\theta(x)) &= \min_\theta \mathbb{E}_{x \sim p(x)} \left[ \ln \frac{p(x)}{p_\theta(x)} \right] \\
&= \min_\theta \mathbb{E}_{x \sim p(x)} [\ln p(x) - \ln p_\theta(x)] \\
&= \min_\theta \underbrace{\mathbb{E}_{x \sim p(x)} [\ln p(x)]}_{\text{not depends on }\theta} - \max_\theta \mathbb{E}_{x \sim p(x)} [\ln p_\theta(x)] \\
&= \max_\theta \mathbb{E}_{x \sim p(x)} [ \ln p_\theta(x) ] \\
&= \mathbb{E}_{x \sim p(x)} [ \max_\theta \ln p_\theta(x) ]
\end{align*}
$$

Minimizing the difference is the same as maximizing the *marginal log-likelihood* $$\ln p_\theta(x)$$ with respect to $$\theta$$,

$$p(x) \approx \max_\theta \ln p_\theta(x)$$

We call this **maximum likelihood estimation**. 

> To maximize a function, we can take the *derivatives*. If $$p_\theta(x)$$ is simply concave or convex, we can easily maximize through $$\displaystyle \frac{d}{d\theta} p_\theta(x) = 0$$

One problem: 

Calculating $$\ln p_\theta(x)$$ is mostly intractable 

* Since the summation or integration happens inside $$\ln$$, we cannot derive a **simple** *closed form expression* for $$\ln p_\theta(x)$$, not to mention the *derivatives* of it.

$$
\begin{align*}
\ln p_\theta(x) &= \ln \left( \sum_z p_\theta(x, z) \right)\\
\text{or } &= \ln \left( \int p_\theta(x, z) dz \right) \quad \text{(for continuous case)}
\end{align*}
$$

* Since the model is a mixture where it is not either concave or convex, it is harder to maximize it with *derivatives*.

We need **approximation** for $$\ln p_\theta(x)$$.

### The Evidence Lower Bound (ELBO)

For simplicity, the following assumes **latent variable** $$z$$ is *continuous* $$z \sim \mathcal{N}(0,1)$$ (though the logic can be applied to *discrete* $$z$$, just turn $$\int$$ to $$\sum$$). This can be helpful when we do **reparameterization** later on.

$$
\begin{gather}
p(z) = \mathcal{N}(z \mid 0,1) \\
p_\theta(x) = \int p_\theta(x, z)dz
\end{gather}
$$

How do we do **approximation** for $$\ln p_\theta(x)$$?

First, let's introduce $$q(z)$$ as in *importance sampling*, where $$q(z)$$ can be any distribution, 

$$p_\theta(x) = \int \frac{q(z)}{q(z)} p_\theta(x, z)dz = \mathbb{E}_{z \sim q(z)} \left[ \frac{p_\theta(x, z)}{q(z)} \right]$$

So the *marginal log-likelihood* $$\ln p_\theta(x)$$,

$$\ln p_\theta(x) = \ln \mathbb{E}_{z \sim q(z)} \left[ \frac{p_\theta(x, z)}{q(z)} \right]$$

Using [**Jensen Inequality**](\post\jensen){:target="_blank"}, we have

$$\ln p_\theta(x) = \ln \mathbb{E}_{z \sim q(z)} \left[ \frac{p_\theta(x, z)}{q(z)} \right] \ge {\color{dodgerblue} \mathbb{E}_{z \sim q(z)} \left[ \ln \frac{p_\theta(x, z)}{q(z)} \right] }$$

We call this the **Evidence Lower Bound (ELBO)**.

We can pick $$q(z)$$ as any distribution. What if $$q(z) = p_\theta(z \mid x)$$?

$$
\begin{align*}
{\color{dodgerblue} \mathbb{E}_{z \sim q(z)} \left[ \ln \frac{p_\theta(x, z)}{q(z)} \right] } &= \mathbb{E}_{z \sim p_\theta(x, z)} \left[ \ln \frac{p_\theta(x, z)}{p_\theta(z \mid x)} \right] \\
&= \mathbb{E}_{z \sim p_\theta(z \mid x)} \left[ \ln \frac{p_\theta(z \mid x)p_\theta(x)}{p_\theta(z \mid x)} \right] \\
&= \mathbb{E}_{z \sim p_\theta(z \mid x)} \left[ \ln p_\theta(x) \right] \\
&= \int p_\theta(z\mid x) \ln p_\theta(x)dz \\
&= \ln p_\theta(x) \underbrace{\int p_\theta(z\mid x) dz}_{1} \\
&= \ln p_\theta(x)
\end{align*}
$$

When we pick $$q(z) = p_\theta(z \mid x)$$, the **ELBO** becomes the *marginal log-likelihood*  $$\ln p_\theta(x)$$ itself. 

Notice $$p_\theta(z \mid x)$$ is actually the *posterior* distribution in **Bayesian Inference**,

$$p_\theta(z \mid x) = \frac{p(z) p_\theta(x \mid z)}{p_\theta(x)}$$

### Variational Inference

We want to find a $$q(z)$$ that can best approximate the *posterior* $$p_\theta(z \mid x)$$, such that **ELBO** can get as close to the *marginal log-likelihood* $$\ln p_\theta(x)$$ itself. This process of using *optimization* as inference is called the **variational inference**. 

To quantify the difference between two distributions $$q(z)$$ and $$p_\theta(z \mid x)$$. We can again use [**KL-divergence**](https://youtu.be/ErfnhcEV1O8){:target="_blank"}.

$$
\begin{align*}
D_{KL} (q(z) \parallel p_\theta(z \mid x)) &= \mathbb{E}_{z \sim q(z)} \left[ \ln \frac{q(z)}{p_\theta(z \mid x)} \right]\\
&= \int q(z) \ln \left( \frac{q(z)}{p_\theta(z \mid x)} \right)dz \\
&= -\int q(z) \ln \left( \frac{p_\theta(z \mid x)}{q(z)} \right)dz\\
&=-\int q(z) \ln \left( \frac{p_\theta(z \mid x)}{q(z)} \frac{p_\theta(x)}{p_\theta(x)} \right)dz\\
&= -\int q(z) \ln \left( \frac{p_\theta(x,z)}{q(z)} \right)dz + \int q(z) \ln p_\theta(x)dz \\
&= -{\color{dodgerblue} \mathbb{E}_{z \sim q(z)} \left[ \ln \frac{p_\theta(x, z)}{q(z)} \right] } + \ln p_\theta(x) \underbrace{\int q(z)dz}_1 \\
&= -{\color{dodgerblue} \mathbb{E}_{z \sim q(z)} \left[ \ln \frac{p_\theta(x, z)}{q(z)} \right] } + \ln p_\theta(x) \ge 0
\end{align*}
$$

> We didn't use $$D_{KL} (p_\theta(z \mid x) \parallel q(z))$$ as we do not know $$p_\theta(z \mid x)$$ and thus we cannot sample $$z$$ from it. See [here](https://ermongroup.github.io/cs228-notes/inference/variational/#on-the-choice-of-kl-divergence){:target="_blank"} for more details on these two choices of **KL-divergence**.

Notice we can derive the **ELBO** again by re-arranging $$D_{KL}$$,

$$\ln p_\theta(x) \ge {\color{dodgerblue} \mathbb{E}_{z \sim q(z)} \left[ \ln \frac{p_\theta(x, z)}{q(z)} \right] }$$

Putting the **ELBO** and $$D_{KL}$$ together, 

$$\ln p_\theta(x) = \text{ELBO} + D_{KL} (q(z) \parallel p_\theta(z \mid x))$$

* The closer $$q(z)$$ gets to $$p_\theta(z \mid x)$$, the lower $$D_{KL}$$. 

* If $$q(z)=p_\theta(z \mid x)$$,  $$D_{KL}=0$$ and **ELBO** becomes the *marginal log-likelihood* $$\ln p_\theta(x)$$.

### Modelling $$q(z)$$

To continue the **variational inference**, we need to model $$q(z)$$. One way to do it is to use a *Neural Network* with weight $$\phi$$, denote as $$q_\phi(z)$$. We want to find a weight $$\phi$$ such that $$q_\phi(z)$$ best approximates the *posterior* $$p_\theta(z \mid x)$$.

A tractable way to do so is to model it using a *Gaussian distribution*

$$q_\phi(z) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi(x))$$

where $$\mu_\phi$$ and $$\sigma_\phi$$ is a function of $$x$$ modelled by the *Neural Network*. Since $$q_\phi(z)$$ depends on $$x$$ and it is used to approximate the *posterior* $$p_\theta(z \mid x)$$, it is better to denote

$$q_\phi(z) = q_\phi(z \mid x)$$

### Maximum Likelihood Estimation with Variational Inference

Now, we have the **ELBO** with $$q_\phi(z) = q_\phi(z \mid x)$$

$${\color{dodgerblue} \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] }$$

Notice the **ELBO** is a function of $$x$$ and takes in TWO weights: $$\theta$$ for $$p_\theta(x,z)$$ and $$\phi$$ for $$q_\phi(z \mid x)$$. For convenience, let's denote the **ELBO** as

$$\mathcal{L}_{\theta, \phi}(x)$$

To comlpete the **variational inference**, we can maximize the **ELBO** with repsect to $$\phi$$, such that $$q_\phi(z \mid x)$$ get as close to the *posterior* $$p_\theta(z \mid x)$$ and thus the **ELBO** get as close to the *marginal log-likelihood* $$\ln p_\theta(x)$$,

$$\ln p_\theta(x) \ge \max_\phi \mathcal{L}_{\theta, \phi}(x)$$


We can then do [**maximum likelihood estimation**](#maximum-likelihood-estimation) by jointly maximize $$\phi$$ and $$\theta$$,

$$\max_\theta \ln p_\theta(x) \ge \max_{\phi, \theta} \mathcal{L}_{\theta, \phi}(x)$$


### Learning with Variational Inference and Stochastic Gradient Descent 

Suppose now we have a dataset $$\mathcal{D}$$ and we can sample $$x \in \mathcal{D}$$. 

Our goal becomes

$$\max_\theta \sum_{x \sim \mathcal{D}} \ln p_\theta(x) \ge \max_{\phi, \theta} \sum_{x \sim \mathcal{D}} \mathcal{L}_{\theta, \phi}(x)$$

Since different sample $$x \in \mathcal{D}$$ might have very different maximized $$\phi$$, things can get so expensive when we have a large dataset $$\mathcal{D}$$. Therefore, it is nice to optimize with [**stochastic gradient descent**](https://youtu.be/Ilg3gGewQ5U?t=575){:target="_blank"}. Suppose we group samples into $$M$$ batches, we denote the batch $$\mathcal{D}_m$$

Our goal for each batch becomes

$$\max_\theta \sum_{x \sim \mathcal{D}_m} \ln p_\theta(x) \ge \max_{\phi, \theta} \sum_{x \sim \mathcal{D}_m} \mathcal{L}_{\theta, \phi}(x)$$

Steps:

1. Initialize $$\theta, \phi$$

2. Randomly sample $$x$$ from the batch $$\mathcal{D}_m$$

3. Optimize $$\phi$$

	* Compute gradient $$\displaystyle \sum_{x \sim \mathcal{D}_m} \nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$

	* Optimize $$\displaystyle \phi \approx \arg \max_\phi \sum_{x \sim \mathcal{D}_m} \mathcal{L}_{\theta, \phi}(x)$$

4. Optimize $$\theta$$

	* Compute gradient $$\displaystyle \sum_{x \sim \mathcal{D}_m} \nabla_\theta \max_\phi \mathcal{L}_{\theta, \phi}(x)$$

	* Update $$\theta$$ according to the gradient direction

5. Repeat from *Step 2* for other batches $$\in \mathcal{D}$$

<br>

As we are *learning* with **variational inference** and **stochastic gradient descent** at the same time, we call this **stochastic variational inference**.

One problem: 

How do we actually compute the gradient $$\nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$ and $$\nabla_\theta \mathcal{L}_{\theta, \phi}(x)$$?

### Monte Carlo Sampling for the ELBO and the gradient $$\nabla_\theta, \nabla_\phi$$

Recall the **ELBO**, 

$$
\begin{align*}
\mathcal{L}_{\theta, \phi}(x) &= {\color{dodgerblue} \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] } \\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln p_\theta(x,z) - \ln q_\phi(z \mid x) \right]
\end{align*}
$$

To compute the **ELBO**, we can first consider deriving a *closed form expression*. In most cases, there is not a simple one. We can use **Monte Carlo sampling** to appoximate it. The idea is to calculate (approximate) the *expectation* using a large number of samples from a distribution. We can sample $$z_1,\dotsc,z_k \sim q_\phi(z \mid x)$$ and estimate

$$\mathcal{L}_{\theta, \phi}(x) \approx \frac{1}{k} \sum_k \ln p_\theta(x,z_k) - \ln q_\phi(z_k \mid x)$$

As $$k$$ is sufficiently large, the approximation is close.

* For gradient $$\nabla_\theta \mathcal{L}_{\theta, \phi}(x)$$, we can easily estimate it using **Monte Carlo sampling**

$$
\begin{align*}
\nabla_\theta \mathcal{L}_{\theta, \phi}(x) &= \nabla_\theta\, \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln p_\theta(x,z) - \ln q_\phi(z \mid x) \right] \\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \nabla_\theta \ln p_\theta(x,z) - \underbrace{\nabla_\theta \ln q_\phi(z \mid x)}_{0} \right] \\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \nabla_\theta \ln p_\theta(x,z) \right] \\
&\approx \frac{1}{k} \sum_k \nabla_\theta \ln p_\theta(x,z_k)
\end{align*}
$$

* For gradient $$\nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$, it is very complicated since the *expectation* depends on $$\phi$$ in $$q_\phi(z \mid x)$$,

$$\nabla_\phi \mathcal{L}_{\theta, \phi}(x) = \nabla_\phi\, \mathbb{E}_{\color{dodgerblue} z \sim q_\phi(z \mid x)} \left[ \ln p_\theta(x,z) - \ln q_\phi(z \mid x) \right]$$

If we want to compute the gradient $$\nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$, we need to make the **ELBO** not an *expectation* depends on $$q_\phi(z \mid x)$$.

> Someone has considered that we can push the gradient $$\nabla_\phi$$ inside the *expectation*. See [here](\post\nabla_phi){:target="_blank"}. But we generally do not use it due to the shortcoming. Instead, we do **reparameterization**.

### Reparameterization

Recall that $$q_\phi(z \mid x)$$ is previously modelled as a *Gaussian distribution*

$$q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi(x))$$

where $$\mu_\phi$$ and $$\sigma_\phi$$ is a function of $$x$$ modelled by the *Neural Network*. 

We can then turn the sampling process from

$$z \sim q_\phi(z \mid x)$$

into 

$$
\begin{gather}
\epsilon \sim \mathcal{N}(0,1) \\
z = {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} = \mu_\phi(x) + \sigma_\phi(x) \epsilon
\end{gather}
$$

Here, we basically map $$\epsilon$$ from 

$$\mathcal{N}(0,1) \to \mathcal{N}(\mu_\phi(x), \sigma_\phi(x))$$

and we turn $$z$$ into a *deterministic* function $${\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}$$. We call this **reparameterization**. 

The **ELBO** *expectation* now no longer depends on $$q_\phi(z \mid x)$$ but $$\mathcal{N}(0,1)$$. 

$$
\begin{align*}
\mathcal{L}_{\theta, \phi}(x) &= \mathbb{E}_{\color{dodgerblue} z \sim q_\phi(z \mid x)} \left[ \ln p_\theta(x,z) - \ln q_\phi(z \mid x) \right] \\
&= \mathbb{E}_{\color{dodgerblue} \epsilon \sim \mathcal{N}(0,1)} \left[ \ln p_\theta(x,{\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} ) - \ln q_\phi({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}\mid x) \right]
\end{align*}
$$

The gradient $$\nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$ becomes

$$
\begin{align*}
\nabla_\phi \mathcal{L}_{\theta, \phi}(x) &= \nabla_\phi\, \mathbb{E}_{\color{dodgerblue} \epsilon \sim \mathcal{N}(0,1)} \left[ \ln p_\theta(x,{\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} ) - \ln q_\phi({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}\mid x) \right] \\
&= \mathbb{E}_{\color{dodgerblue} \epsilon \sim \mathcal{N}(0,1)} \left[ \nabla_\phi \ln p_\theta(x,{\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} ) - \nabla_\phi \ln q_\phi({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}\mid x) \right] 
\end{align*}
$$

We can now approximate it using **Monte Carlo sampling** again. We sample $$\epsilon_1 \dotsc, \epsilon_k \sim \mathcal{N}(0,1)$$,

$$\nabla_\phi \mathcal{L}_{\theta, \phi}(x) \approx \frac{1}{k} \sum_k \nabla_\phi \ln p_\theta(x,{\color{dodgerblue} \mathcal{G}_\phi(\epsilon_k, x)} ) - \nabla_\phi \ln q_\phi({\color{dodgerblue} \mathcal{G}_\phi(\epsilon_k, x)}\mid x)$$


<hr>

<p class="centre"> part 2 </p>

<hr>

### Variational Auto-encoder (VAE)

Now, let's move on to the model architecture point of view. Consider the following *Neural Network*,

![vae](\assets\img\vae-architecture.png "vae")

* The **encoder** models $$q_\phi(z \mid x)$$  
(Remember we hope $$q_\phi(z \mid x)$$ to get as close to the *posterior* $$p_\theta(z \mid x)$$)

	* Input: image $$x$$ 

	* Output: $$\mu_\phi(x), \sigma_\phi(x)$$ for the **latent** distribution  

* The **decoder** models $$p_\theta(x \mid z)$$ 

	* Input: sample $$z$$ from the **latent** distribution with $$\mu_\phi(x), \sigma_\phi(x)$$

	* Output: reconstruction image $$\tilde{x}$$

The **decoder** is a **generative model** since we can learn a probability distribution $$p_\theta(x \mid z)$$ and sample $$\tilde{x}$$ from it. 

### Objective function

There are different goals for such an architecture. Possible goals can be **de-noising** or **inpainting**. Here, our goal is to have the reconstruction image $$\tilde{x}$$ as close to the original image $$x$$, so we want our $$p_\theta(x)$$ get as close to the real distribution $$p(x)$$. 

How? [**Maximum Likelihood Estimation**](#maximum-likelihood-estimation-with-variational-inference)!

$$\max_\theta \ln p_\theta(x) \ge \max_{\phi, \theta} \mathcal{L}_{\theta, \phi}(x)$$

We encounter the **ELBO** again as our **objective function**! 

$$\mathcal{L}_{\theta, \phi}(x) = {\color{dodgerblue} \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] }$$

But notice the **ELBO** doesn't have a term $$p_\theta(x \mid z)$$ which is the **decoder** part, we need to do something with it, 

$$
\begin{align*}
\mathcal{L}_{\theta, \phi}(x) &= {\color{dodgerblue} \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \right] } \\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{p_\theta(x \mid z)p(z)}{q_\phi(z \mid x)} \right]\\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln p_\theta(x \mid z) + \ln \frac{p(z)}{q_\phi(z \mid x)} \right]\\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln p_\theta(x \mid z)] - \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{q_\phi(z \mid x)}{p(z)} \right]\\
&= \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder}] - D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z))
\end{align*}
$$

The **ELBO** now becomes **VERY NICE** since it captures only 

* the **encoder** part $$q_\phi(z \mid x)$$

$$q_\phi(z \mid x) = \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi(x))$$  

* the **decoder** part $$p_\theta(x \mid z)$$

$$p_\theta(x \mid z) = \mathcal{N}(z \mid \mu_\theta(x), \sigma_\theta(x))$$

* the distribution for the **latent variables** $$p(z)$$  

$$p(z) = \mathcal{N}(z \mid 0,1)$$

Our objective is to maximize the **ELBO**, we can maximize both terms,

$$\max_{\phi, \theta} \mathcal{L}_{\theta, \phi}(x) = \max_{\phi, \theta} \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder} ] + \max_\phi -D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z))$$

> We call this $$D_{KL}$$ the *exclusive* **KL-divergence** or the I-projection. Notice when $$p(z)=0$$ and $$q_\phi(z \mid x) \ge 0$$, then $$D_{KL} \to \infty$$. Since we want $$D_{KL} \to 0$$, so wherever $$p(z)=0$$, we will push $$q_\phi(z \mid x)=0$$. See [here](https://ermongroup.github.io/cs228-notes/inference/variational/#on-the-choice-of-kl-divergence){:target="_blank"} for more details.

<br>

We can then do [**stochastic variational inference**](#learning-with-variational-inference-and-stochastic-gradient-descent) with the gradient $$\nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$ and $$\nabla_\theta \mathcal{L}_{\theta, \phi}(x)$$. As we are doing **variational inference**, that's why we have the name **"variational" auto-encoder**.


### Computing the gradient $$\nabla_\phi \mathcal{L}_{\theta, \phi}(x)$$ and $$\nabla_\theta \mathcal{L}_{\theta, \phi}(x)$$

Let's handle the gradient $$\nabla_\phi$$ first,

$$\nabla_\phi \mathcal{L}_{\theta, \phi}(x) = 
\nabla_\phi\, \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder} ] + \nabla_\phi -D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z))$$

Notice $$\phi$$ affects BOTH the **encoder** and **decoder** part.

We denote 

* the **decoder** (first) part as $$\nabla_\phi\, \mathbb{E}$$ 

* the **encoder** (second) part as $$\nabla_\phi -D_{KL}$$

Then the gradient $$\nabla_\theta$$,

$$
\begin{align*}
\nabla_\theta \mathcal{L}_{\theta, \phi}(x) &= \nabla_\theta\, \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder} ] + \nabla_\theta \underbrace{ -D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z)) }_{\text{not depends on } \theta} \\
&= \nabla_\theta\, \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln p_\theta(x \mid z) ] \\
\end{align*}
$$

Notice $$\theta$$ only affects the **decoder** part. 

We denote

* this **decoder** part as $$\nabla_\theta\, \mathbb{E}$$



In short, we have three gradients to handle, 

* **encoder** part: $$\nabla_\phi -D_{KL}$$

* **decoder** part: $$\nabla_\phi\, \mathbb{E}$$ and $$\nabla_\theta\, \mathbb{E}$$


### Gradient for the encoder part $$\nabla_\phi -D_{KL}$$

The gradient for the **encoder** part, 

$$
\begin{align*}
\nabla_\phi -D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z)) &=
\nabla_\phi -\mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln \frac{q_\phi(z \mid x)}{p(z)} \right] \\
&= \nabla_\phi -\int q_\phi(z \mid x) \ln \frac{q_\phi(z \mid x)}{p(z)} dz 
\end{align*}
$$

Recall that $$p(z)$$ and $$q_\phi(z \mid x)$$ are treated as *Gaussian distributions*. 

$$
\begin{align*}
p(z) &= \mathcal{N}(z \mid 0,1) \\
q_\phi(z \mid x) &= \mathcal{N}(z \mid \mu_\phi(x), \sigma_\phi(x))
\end{align*}
$$

For simplicity, the following simply use $$\mu, \sigma$$ instead of $$\mu_\phi(x), \sigma_\phi(x)$$. 

$$-D_{KL}$$ is essentially doing

$$
\begin{align*}
-D_{KL}(q_\phi(z \mid x) \parallel p(z)) &= -D_{KL}(\mathcal{N}(z \mid \mu, \sigma) \parallel \mathcal{N}(z \mid 0,1) ) \\
&= -\int \mathcal{N}(z \mid \mu, \sigma) \ln \frac{\mathcal{N}(z \mid \mu, \sigma)}{\mathcal{N}(z \mid 0,1)}dz 
\end{align*}
$$

Shall we go straight to [**Monte Carlo sampling**](#monte-carlo-sampling-for-the-elbo-and-the-gradient-nabla_theta-nabla_phi)? In fact, there is a *closed form expression* for this *integral*.

Plugging in the [**Gaussian PDF**](\post\gaussian){:target="_blank"},

$$
\begin{align*}
\ln \frac{\mathcal{N}(z \mid \mu, \sigma)}{\mathcal{N}(z \mid 0,1)} &= \ln \left( \cfrac{ \cfrac{1}{\sqrt{2\pi}\sigma} \, e^{\,-\cfrac{(z-\mu)^2}{2\sigma^2}} }{ \cfrac{1}{\sqrt{2\pi}} \, e^{\,-\cfrac{(z)^2}{2}} } \right) \\
&= \ln \left( \frac{1}{\sigma} \cdot e^{\,-\frac{(z-\mu)^2}{2\sigma^2} +\frac{(z)^2}{2}} \right) \\
&= -\ln\sigma + \left( -\frac{(z-\mu)^2}{2\sigma^2} +\frac{(z)^2}{2} \right) \\
&= -\frac{1}{2} \ln(\sigma^2) +\frac{1}{2} \left( z^2 -\frac{(z-\mu)^2}{\sigma^2} \right) \\ 
&= \frac{1}{2} \left( -\ln(\sigma^2) + z^2 -\frac{(z-\mu)^2}{\sigma^2} \right)
\end{align*}
$$

Altogether we have

$$
\begin{align*}
&\, -\int \mathcal{N}(z \mid \mu, \sigma) \ln \frac{\mathcal{N}(z \mid \mu, \sigma)}{\mathcal{N}(z \mid 0,1)}dz \\
&= -\frac{1}{2} \int \left( \frac{1}{\sqrt{2\pi}\sigma} \, e^{\,-\frac{(z-\mu)^2}{2\sigma^2}} \right) 
\left( -\ln(\sigma^2) + z^2 -\frac{(z-\mu)^2}{\sigma^2} \right)dz 
\end{align*}
$$

> Note that if we do [Reparameterization](#reparameterization) here to sample $$\epsilon \sim \mathcal{N}(0,1)$$ and map $$\epsilon \to z=\mu+\sigma\epsilon$$ for $$-D_{KL}$$, we can still derive the same *closed-form expression. See [here](\post\epsilon){:target="_blank"} for details.*

<br>

The definite integral for each part,

$$
\begin{align*}
\int_{-\infty}^{+\infty} \left( \frac{1}{\sqrt{2\pi}\sigma} \, e^{\,-\frac{(z-\mu)^2}{2\sigma^2}} \right)
\left( -\ln(\sigma^2) \right)dz &= -\ln(\sigma^2) \\
\int_{-\infty}^{+\infty} \left( \frac{1}{\sqrt{2\pi}\sigma} \, e^{\,-\frac{(z-\mu)^2}{2\sigma^2}} \right)
\left( z^2 \right)dz &= \mu^2 + \sigma^2 \\
\int_{-\infty}^{+\infty} \left( \frac{1}{\sqrt{2\pi}\sigma} \, e^{\,-\frac{(z-\mu)^2}{2\sigma^2}} \right)
\left( -\frac{(z-\mu)^2}{\sigma^2} \right)dz &= -1 \\
\end{align*}
$$

Finally, we have a *closed form expression*

$$-D_{KL}(q_\phi(z \mid x) \parallel p(z)) = \frac{1}{2} ( 1 + \ln(\sigma^2) - \mu^2 - \sigma^2 )$$

Notice we can view it as 2 parts to maximize,

$$\max_\phi -D_{KL}(q_\phi(z \mid x) \parallel p(z)) = \frac{1}{2} ( \underbrace{1 + \ln(\sigma^2) - \sigma^2}_{\max_\phi} \underbrace{- \mu^2}_{\max_\phi})$$

See the graph in the following [**Desmos**](https://www.desmos.com/calculator){:target="_blank"} API, 

* $$\displaystyle \max_\phi (1 + \ln (\sigma^2) - \sigma^2)$$ pushes $$\sigma^2$$ to 1 

* $$\displaystyle \max_\phi -\mu^2$$ pushes $$\mu$$ to 0 

This is exactly what we are aiming to do in $$\max -D_{KL}$$, we are pushing $$q_\phi(z \mid x)$$ closer to $$p(z) = \mathcal{N}(z \mid 0,1)$$

<div id="calculator 1" style="width: 600px; height: 400px; margin-left: auto;
    margin-right: auto; margin-top: 2rem; margin-bottom: 3rem;"></div>


The gradient $$\nabla_\phi$$

$$
\begin{align*}
\nabla_\phi -D_{KL}(q_\phi(z \mid x) \parallel p(z)) &= \nabla_\phi \ln(\sigma^2) - \nabla_\phi\, \mu^2 - \nabla_\phi\, \sigma^2 \\
&= \frac{2}{\sigma} (\nabla_\phi\, \sigma) - (2\mu) (\nabla_\phi\, \mu) - (2\sigma) (\nabla_\phi\, \sigma) \\
&= 2\left( \frac{1}{\sigma} - \sigma \right) (\nabla_\phi\, \sigma) - (2\mu) (\nabla_\phi\, \mu)
\end{align*}
$$

where

$$
\begin{align*}
\nabla_\phi\, \mu &= \nabla_\phi\, \mu_\phi(x) \\
\nabla_\phi\, \sigma &= \nabla_\phi\, \sigma_\phi(x)
\end{align*}
$$

Note that $$\mu_\phi(x)$$ and $$\sigma_\phi(x)$$ are the outputs of the **encoder** part. These two gradients can be computed in the *Neural Network* using [**back-propagation**](https://youtu.be/tIeHLnjs5U8){:target="_blank"}.

### Gradient for the decoder part $$\nabla_\phi\, \mathbb{E}$$ and $$\nabla_\theta\, \mathbb{E}$$

The gradient for the **decoder** part,

$$\nabla \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder} ]$$

Notice the *expectation* depends on $$\phi$$ in $$q_\phi(z \mid x)$$, so we cannot easily compute the gradient $$\nabla_\phi$$. How to compute then? 

[**Reparameterization**](#reparameterization) and [**Monte Carlo sampling**](#monte-carlo-sampling-for-the-elbo-and-the-gradient-nabla_theta-nabla_phi) as mentioned above! We can do it on both $$\nabla_\phi$$ and $$\nabla_\theta$$ for convenience.

$$
\begin{align*}
\nabla_\phi\, \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln p_\theta(x \mid z)] &= 
\nabla_\phi\, \mathbb{E}_{\color{dodgerblue} \epsilon \sim \mathcal{N}(0,1)} [ \ln p_\theta(x \mid{\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)})] \\
&\approx \frac{1}{k} \sum_k \nabla_\phi \ln p_\theta(x \mid{\color{dodgerblue} \mathcal{G}_\phi(\epsilon_k, x)}) \\
\nabla_\theta\, \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln p_\theta(x \mid z)] &= 
\nabla_\theta\, \mathbb{E}_{\color{dodgerblue} \epsilon \sim \mathcal{N}(0,1)} [ \ln p_\theta(x \mid{\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)})] \\
&\approx \frac{1}{k} \sum_k \nabla_\theta \ln p_\theta(x \mid{\color{dodgerblue} \mathcal{G}_\phi(\epsilon_k, x)})
\end{align*}
$$

where

$$
\begin{gather}
\epsilon_1 \dotsc, \epsilon_k \sim \mathcal{N}(0,1) \\
z = {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} = \mu_\phi(x) + \sigma_\phi(x) \epsilon
\end{gather}
$$

### Reparameterization for the decoder

So now, we DO NOT want to sample $$z$$ from the **latent** distribution as input for the **decoder**, instead we want to sample $$\epsilon \sim \mathcal{N}(0,1)$$

* The **decoder** models $$p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)})$$ 

	* Input: sample $$\epsilon$$ from $$\mathcal{N}(0,1)$$ and map it to $$z$$ with a *deterministic function* $$z = {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} = \mu_\phi(x) + \sigma_\phi(x) \epsilon$$

	* Output: reconstruction image $$\tilde{x}$$

Turning $$z$$ into a *deterministic function* allows us to do [**back-propagation**](https://youtu.be/tIeHLnjs5U8){:target="_blank"} easily.

![reparam](\assets\img\vae_reparam.png "Reparameterization for VAE")


### Computing the decoder log-likelihood $$\ln p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)})$$

Recall very early in [**Gaussian Mixture model**](#gaussian-mixture-model), we assume

$$p_\theta(x \mid z) = \mathcal{N}(x \mid \mu_\theta(z), \sigma_\theta(z))$$

Now with $$z = {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} = \mu_\phi(x) + \sigma_\phi(x) \epsilon$$, 

$$p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) = \mathcal{N}(x \mid \mu_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}), \sigma_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}))$$

We can compute the *log-likelihood*,

$$
\begin{align*}
\ln p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) &= \ln \mathcal{N}(x \mid \mu_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}), \sigma_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)})) \\
&= \ln \left( \frac{1}{\sqrt{2\pi}\sigma_\theta} \, e^{\,-\frac{(x-\mu_\theta)^2}{2\sigma_\theta^{\,\,2}}} \right) \\

&= - \frac{\ln(2\pi\sigma_\theta^{\,\,2})}{2} -\frac{(x-\mu_\theta)^2}{2\sigma_\theta^{\,\,2}} \\
&= \frac{1}{2} \left[ -\ln(2\pi\sigma_\theta^{\,\,2}) -\frac{1}{\sigma_\theta^{\,\,2}} (x-\mu_\theta)^2\right]
\end{align*}
$$

As we want to maximize the *log-likelihood*,

$$
\max \ln p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) = 
\frac{1}{2} \left[ 
- \underbrace{ \ln(2\pi\sigma_\theta^{\,\,2}) }_{\min}
- \underbrace{ \frac{1}{\sigma_\theta^{\,\,2}}  
\underbrace{ (x-\mu_\theta)^2 }_{\min} }_{\min}
\right]
$$

Let $$k = x-\mu_\theta$$ and plot the whole function in a 2D plane with x-coordinate as $$\sigma_\theta^{\,\,2}$$. 

* As $$k \to 0$$, x-coordinate $$\sigma_\theta^{\,\,2}$$ is being pushed to 0 

You can play with the slider $$k$$ in the following [**Desmos**](https://www.desmos.com/calculator){:target="_blank"} API,

<div id="calculator 2" style="width: 600px; height: 400px; margin-left: auto;
    margin-right: auto; margin-top: 2rem; margin-bottom: 3rem;"></div>


We do not want to push $$\sigma_\theta \to 0$$. 

A good method to avoid that is to lower bound $$\sigma_\theta$$ to 1. In practice, we will simply set the variance $$\sigma_\theta$$ to 1 (as *hyperparameters*) and DO NOT learn it in the **decoder**.  

Now, we have a *closed form expression* with $$\sigma_\theta=1$$,

$$
\begin{align*}
\ln p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) &= \ln \mathcal{N}(x \mid \mu_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}), \sigma_\theta=1) \\
&= - \frac{\ln(2\pi)}{2} -\frac{(x-\mu_\theta)^2}{2} 
\end{align*}
$$

where 

$$\mu_\theta = \mu_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) = \mu_\theta({\color{dodgerblue} \mu_\phi(x)+\sigma_\phi(x)\epsilon})$$

Since we have set $$\sigma_\theta=1$$, our **decoder** network only learns $$\theta$$ for $$\mu_\theta$$ and produce the output $$\tilde{x}$$. So we can simply treat $$\mu_\theta$$ as the reconstruction output image $$\tilde{x}$$.

The term 

$$-\frac{(x-\mu_\theta)^2}{2} = -\frac{(x-\tilde{x})^2}{2}$$ 

is simply a negative **mean-squared error**, which is a very common **loss function**. We can also call it **reconstruction loss** since it is calculating the difference between reconstruction output $$\tilde{x}$$ and original input $$x$$.

For the gradient $$\nabla_\phi$$ or $$\nabla_\theta$$, what's left is

$$
\begin{align*}
\nabla_\theta \ln p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) &= -(x - \mu_\theta) \cdot \nabla_\theta \mu_\theta \\
\nabla_\phi \ln p_\theta(x \mid {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)}) 
&= -(x - \mu_\theta) \cdot \nabla_\phi \mu_\theta \\
&= -(x - \mu_\theta) \cdot \nabla_{\color{dodgerblue} \mathcal{G}} \mu_\theta \cdot 
\nabla_\phi {\color{dodgerblue} \mathcal{G}_\phi(\epsilon, x)} \\
&= -(x - \mu_\theta) \cdot \nabla_{\color{dodgerblue} \mathcal{G}} \mu_\theta \cdot 
\nabla_\phi ({\color{dodgerblue} \mu_\phi(x)+\sigma_\phi(x)\epsilon}) \\
&= -(x - \mu_\theta) \cdot \nabla_{\color{dodgerblue} \mathcal{G}} \mu_\theta \cdot 
(\nabla_\phi \mu_\phi(x) + \epsilon \nabla_\phi \sigma_\phi(x))
\end{align*}
$$

Note that we have already encountered $$\nabla_\phi \mu_\phi(x)$$ and $$\nabla_\phi \sigma_\phi(x)$$ when we are handling [$$\nabla_\phi -D_{KL}$$](#gradient-for-the-encoder-part-nabla_phi--d_kl). We can compute them, along with $$\nabla_\theta \mu_\theta$$, using [**back-propagation**](https://youtu.be/tIeHLnjs5U8){:target="_blank"}.

### Loss Function

After handling all the gradients $$\nabla$$, let's finish up things. 

Remember we are maximizing the *ELBO* for $$\ln p_\theta(x)$$ as our [**objective function**](#objective-function),

$$\max_{\phi, \theta} \mathcal{L}_{\theta, \phi}(x) = \max_{\phi, \theta} \mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder} ] + \max_\phi -D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z))$$

Normally, we want to find a **loss function** to minimize. We can easily turn the **ELBO** into a minimizing case,

$$\min_{\phi, \theta} -\mathcal{L}_{\theta, \phi}(x) = \min_{\phi, \theta} -\mathbb{E}_{z \sim q_\phi(z \mid x)} [ \ln \underbrace{p_\theta(x \mid z)}_\text{decoder} ] + \min_\phi D_{KL}(\underbrace{q_\phi(z \mid x)}_\text{encoder} \parallel p(z))$$

With the *closed form expressions* we have deduced, we can re-write

$$
\min_{\phi, \theta} -\mathcal{L}_{\theta, \phi}(x) \approx 
\min_{\phi, \theta} 
\underbrace{\frac{1}{k} \sum_k \frac{(x-\tilde{x})^2}{2}}_\text{decoder loss} + 
\min_\phi 
\underbrace{\frac{1}{2} ( \mu^2 + \sigma^2 - 1 -\ln(\sigma^2))}_\text{encoder loss}
$$

where we have [**reparameterization for the decoder**](#reparameterization-for-the-decoder),

$$
\begin{gather}
\epsilon_1 \dotsc, \epsilon_k \sim \mathcal{N}(0,1) \\
\tilde{x} = \mu_\theta({\color{dodgerblue} \mathcal{G}_\phi(\epsilon_k, x)}) = \mu_\theta({\color{dodgerblue} \mu_\phi(x)+\sigma_\phi(x)\epsilon_k})
\end{gather}
$$

In common, 

* the **decoder loss** is often named as **reconstruction loss**

* the **encoder loss** is often named simply as **KL-divergence** and treated as the **regularization term**

It is great to keep in mind what they are actually doing in terms of the architectural point of view, and also connecting them to the [**statistical concepts**](#wrapping-the-statistical-concepts) fundamentally.

<hr>

<p class="centre"> We can implement now ~.~ </p>

<hr>

### VAE with Python / Keras

See [**Code example**](https://keras.io/examples/generative/vae/){:target="_blank"}

### Pros and Cons

* Pros

	* Flexible

	* Generative

* Cons

	* Need approximation for *marginal log-likelihood*  
	(**Variational Inference**) 

	* Learning (optimizing) is hard

### Experiments

#TODO

### Related Applications

* VQ-VAE  
(*discrete* latent variables instead of *continuous*)

### Further Research / Improvement

* More expressive approximating families  
(other than [**Gaussian Mixture model**](#gaussian-mixture-model))

* Better learning (optimizing) algorithm  
(other than [**stochastic variational inference**](#learning-with-variational-inference-and-stochastic-gradient-descent))  
(i.e. ADAM?)

* Better objective function  
(other than [**ELBO**](#objective-function))


<script>
  var elt = document.getElementById('calculator 1');
  var calculator1 = Desmos.GraphingCalculator(elt);

  calculator1.setExpression({ 
  	id: 'graph1', 
  	latex: 'y= 1 + \\ln x - x',
  	color: Desmos.Colors.RED
  });

  calculator1.setExpression({ 
  	id: 'graph2', 
  	latex: 'y= - \\sqrt{x^2}',
  	color: Desmos.Colors.BLUE
  });


  var elt = document.getElementById('calculator 2');
  var calculator2 = Desmos.GraphingCalculator(elt);

  calculator2.setExpression({ 
  	id: 'graph1', 
  	latex: 'y=-\\ln (2 \\pi x) - {k^2}/x',
  	color: Desmos.Colors.RED
  });

  calculator2.setExpression({ 
  	id: 'slider', 
  	latex: 'k=1',
  	sliderBounds: { min: 0, max: 2, step: '' }
  });
</script>
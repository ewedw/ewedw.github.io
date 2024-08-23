---
layout: post
title:  Jensen Inequality
date: 2022-08-20 05:00:00 +0800
permalink: /post/jensen/

---

# Jensen Inequality

Consider the following graph with $$f(x)$$ and $$q(x)$$, 

![jensen](\assets\img\jensen-1.png "f(x) > q(x)")

Since this is a post related to probabilistic models which have extensive use of *log-likelihood*, we use $$f(x) = \ln(x)$$. It is *concave* as shown above. 

Intuitively, if we introduce a *secent line* $$q(x)$$ that cuts through $$f(x)$$ at $$x_1$$ and $$x_2$$, within that range, 

* $$f(x) \ge q(x)$$ when $$f(x)$$ is *concave* (this case)

* $$f(x) \le q(x)$$ when $$f(x)$$ is *convex*

It is obvious when we look at the above graph. 

### Jensen Inequality

Jensen provides us an *algebraic expression* to describe the above case, which can be very helpful when we are doing *optimization*. The *Evidence Lower Bound (ELBO)* in [**variational inference**](\post\latent\#variational-inference){:target="_blank"} is a great example.

![jensen](\assets\img\jensen-2.png "Jensen Inequality")

Let $$x_1 \le x_k \le x_2$$ and $$f$$ is a *concave* function, we have

$$f(x_k) \ge q(x_k)$$

Let the distance between $$x_1$$ and $$x_k$$ take a ratio $${\color{orange}t}$$, the distance between $$x_k$$ and $$x_2$$ then takes a ratio $${\color{orange}1-t}$$. Using *section formula*,

$$x_k = {\color{orange}(1-t)}x_1 + {\color{orange}(t)}x_2$$

Since $${\color{orange}f(x_1)}, q(x_k), {\color{orange}f(x_2)}$$ are on the same *secent line* $$q(x)$$, we can also express $$q(x_k)$$ with *section formula*, 

$$q(x_k) = (1-t){\color{orange}f(x_1)} + (t){\color{orange}f(x_2)}$$

Combining the above three expressions, we have the **Jensen Inequality**, given $$f$$ is *concave* within $$[x_1,x_2]$$,

$$
\begin{align*}
f(x_k) &\ge q(x_k) \\
f((1-t)x_1 + (t)x_2) &\ge (1-t)f(x_1) + (t)f(x_2)
\end{align*}
$$

As in *probability theory*, we can write it as *expected value*,

$$f(\mathbb{E}[X]) \ge \mathbb{E}[f(X)]$$

where $$f$$ denotes a *concave function*.

### Jensen Inequality with Log-likelihood 

As we have $$f(x) = \ln(x)$$, 

$$\ln(\mathbb{E}[X]) \ge \mathbb{E}[\ln(X)]$$

*Log-likehood* takes the form of $$\ln P(x)$$, we can set $$X=P(x)$$, then

$$\ln(\mathbb{E}[P(x)]) \ge \mathbb{E}[\ln P(x)]$$

Suppose we have $$p_{\theta} (x, z)$$, we can marginalize $$z$$ to find $$p_{\theta} (x)$$, 

$$p_{\theta} (x) = \sum_z p_{\theta} (x, z)$$

By multiplying $$\displaystyle\frac{q(z)}{q(z)}$$, we can turn it into an *expected value* with respect to $$q(z)$$,

$$p_{\theta} (x) = \sum_z \frac{q(z)}{q(z)} p_{\theta} (x, z) = \mathbb{E}_{z \sim q(z)} \left[ \frac{p_{\theta} (x, z)}{q(z)} \right]$$

So the *log-likelihood* $$\ln p_{\theta} (x)$$, 

$$\ln p_\theta(x)= \ln \left( \mathbb{E}_{z \sim q(z)} \left[ \frac{p_{\theta} (x, z)}{q(z)} \right] \right)$$

Using the **Jensen Inequality**, we actually derive the *Evidence Lower Bound (ELBO)* mentioned earlier for *log-likelihood* $$\ln p_{\theta} (x)$$,

$$\ln p_\theta(x) = \ln \left( \mathbb{E}_{z \sim q(z)} \left[ \frac{p_{\theta} (x, z)}{q(z)} \right] \right) \ge \mathbb{E}_{z \sim q(z)} \left[ \ln \left( \frac{p_{\theta} (x, z)}{q(z)} \right) \right]$$
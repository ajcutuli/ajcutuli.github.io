---
layout: archive
permalink: /HMMVI/
header-includes:
    - \usepackage{bbm}
author_profile: true
title: "Pre-conditioning Hamiltonian Monte Carlo with Variational Inference"
---

{% include base_path %}
### Abstract


## Introduction
A core problem in modern statistics is dealing with the computational complexity and challenges associated with calculating high-dimensional probability distributions.
This is of immense interest in Bayesian inference, where the analytical evaluation of interesting models is often rendered infeasible due to complex marginal likelihoods.
This necessitates the use of numerical methods such as Markov chain Monte Carlo (MCMC), which iteratively runs a Markov chain with the goal of generating samples from a target distribution.
However, traditional MCMC follows a random walk, allowing complex posterior geometries to trap samplers in local modes and prevent them from exploring other significant areas of the parameter space.
A particular class of techniques that has achieved recognized success in accelerating convergence in high-dimensional models is Hamiltonian Monte Carlo (HMC; [Neal, 2012](#Neal); [Betancourt, 2018](#HMC)), which uses gradients of the log-posterior to propose movements across the parameter space.
These sampling algorithms are asymptotically exact, but tuning them to ensure efficient convergence can be a difficult task.

One component of HMC that complicates its implementation is the initialization of a *mass matrix* that helps determine how efficiently the sampler explores different Hamiltonian values and hence the parameter space.
If the mass matrix does not align well with the geometry of the distribution, the sampler can become inefficient, making slow progress or requiring many more steps to explore the space adequately.
In low-dimensional settings, it's generally accepted to approximate the mass matrix by the precision matrix of the target distribution.
In high-dimensional settings, however, inverting a covariance matrix can be computationally infeasible, and so most scalable implementations of HMC initialize the mass matrix to be diagonal.

However, there is no clear concensus on what to set the diagonal elements to.
Most state-of-the-art HMC implementations take the diagonal elements to be early estimates of the auxiliary momentum variables' marginal variances ([Hoffman and Gelman, 2014](#nouturn)), but [Tran and Kleppe (2024)](#Tran) recently concluded that perhaps marginal precision estimates may be a good choice for a variety of problems.
Meanwhile, [Margossian et al. (2024)](#VI-tradeoffs) suggest that variational inference (VI; [Blei et al., 2017](#VI)) may be used to tune the mass matrix.
By minimizing a divergence measure between the true momentum distribution and an approximation involving a diagonal mass matrix, we can generate a black-box suggestion for the mass matrix.
In this post, we evaluate the efficacy of this idea for high-dimensional models.

## Background

### Hamiltonian Monte Carlo
Markov chain Monte Carlo (MCMC) aims to sample from a distribution over parameters $q\in \mathbb{R}^d$ with probability density function $f(q)$ by building a Markov chain where each step is a new "guess" or sample of the parameter values, generated based on the previous sample.
Over time, this chain is designed to traverse regions of high density (e.g. neighborhoods of the mode) in an effort to reflect the desired target distribution.
A common application of MCMC is to estimate statistics for this target distribution (e.g. mean, variance, etc.), which requires integrating the target distribution over the volume of parameter space.
Now, when $d$ is large, the volume contribution of high-density regions decreases exponentially and more volume becomes concentrated in low-density tails.
As a result, traditional MCMC samplers that follow a random walk tend to become "stuck" exploring high-density regions while ignoring regions of higher probability mass.
Hence, retrieving samples that accurately integrate the density across the entire parameter space becomes very inefficient if we rely on uninformed transition criteria.

To combat this, Hamiltonian Monte Carlo (HMC; [Neal, 2012](#Neal); [Betancourt, 2018](#HMC)) utilizes Hamiltonian dynamics to inform the direction of Markov transitions by exploiting knowledge regarding the geometry of the target distribution.
If tuned properly, the sampling algorithm guarantees exploration of the *typical set* (i.e. the regions between the tails and the modes with higher probability mass).
HMC operates by lifting *position* $q$ into a $2d$-dimensional *phase space* $(q,p)$, where $p\in \mathbb{R}^d$ is an auxiliary *momentum* variable.
The joint probability distribution in phase space is the *canonical distribution*
$$
\pi(q,p)=\pi(p\mid q)\pi(q),
$$
which marginilizes out such that the target distribution $f(q)$ can be mapped back to.
The canonical density can be written as a Boltzmann distribution in terms of a *Hamiltonian* function that characterizes the value—or *energy*—of any point in the phase space:
$$
\pi(q,p)=\exp(-H(q,p)).
$$
The Hamiltonian may be decomposed as
$$
\begin{aligned}
H(q,p)=&~-\log \pi(q,p)\\
=&~-\log \pi(p\mid q)-\log \pi(q)\\
=&~K(q,p)+U(q),
\end{aligned}
$$
where $K(q,p)$ is a *kinetic energy* function and $U(q)$ is a *potential energy* function but also the negative logarithm of the target distribution.

With the introduction of $K(q,p)$ and $U(q)$, a physical analogy for the system nicely emerges.
Consider a flat frictionless surface in which a point mass travels at a constant velocity. When it encounters an upward slope, its momentum carries it forward, converting kinetic energy into potential energy. It continues up the slope until all kinetic energy is converted into potential energy, then reverses direction, sliding back down as potential energy decreases and kinetic energy increases.
These dynamics are described by *Hamilton's equations*:
$$
\begin{aligned}
\dot{q}(t)=&~+\nabla_p H=\nabla_p K\\
\dot{p}(t)=&~-\nabla_q H=-\nabla_q K - \nabla_q U.
\end{aligned}
$$

Since $- \nabla_q U$ is the gradient of the logarithm of the target distribution, we see that integrating Hamiltonian dynamics over time and mapping back to parameter space gives a Markov transition that is informed by the geometry of the target distribution.
<!-- In fact, if the random lift results in a value in the typical set of phase space, then the mapping back to parameter space results in a value $ -->

Moreover, if an initial point $q$ lies in the typical set of the target distribution, then sampling
$$
p\sim \pi(p\mid q)
$$
results in a point $(q,p)$ that sits in an energy level of the typical set of phase space.
Now, Hamiltonian dynamics preserve the total energy of the physical system, so integrating Hamilton's equations over time $t$ gives an image $\phi_t(q,p)=(\tilde q,\tilde p)$ that is also in the typical set, and $\tilde q$ is in the typical set of the target distribution.
Put simply, a well-configured HMC implementation doesn't lead the sampler out of the typical set and into regions of low probability mass.
Ensuring this conservation requires that any transformation of parameter space be complemented by an inverse transformation in momentum space.

#### Optimizing kinetic energy
Implementing HMC in an optimal way requires carefully tuning the kinetic energy function and the integration time of each Hamiltonian mapping.
An ideal kinetic energy function is critical to ensuring that momentum sampling explores all possible energy levels of the typical set, while the integration time determines how well the Hamiltonian mapping explores the phase space of a given energy level.

Choosing integration time generally relies on heuristic stopping criteria involving step size $\epsilon$ and an adaptive step count $L$ that characterize how Hamilton's equations are discretized ([Hoffman and Gelman, 2014](#nouturn)).
In this post we focus primarily on the choice of kinetic energy function which defines $\pi(p\mid q)$.

One common choice for the conditional momentum distribution is
$$
\pi(p\mid q)=\mathcal{N}(0,M),
$$
which is equivalent to kinetic energy function
$$
K(q,p)=\frac{1}{2}p^TM^{-1}p+\log|M|+\text{const}.
$$
This $M$ is referred to as the *mass matrix*, which can be used to rotate and scale parameter space such that momentum sampling is more uniform and hence energy levels can be more efficiently explored.

In particular, since position and momentum are inversely related, a transformation $\tilde p=\sqrt{M^{-1}}p$ is complemented by a transformation $\tilde q=\sqrt{M}(q-\mu)$, where $\mu$ helps center the target about $0$.
Such a transformation effectively de-correlates momentum space by simplifying the kinetic energy to
$$
K(\tilde q,\tilde p)=\frac{1}{2}(\tilde p)^T\tilde p+\log|M|+\text{const}.
$$

Accordingly, Hamilton's equations can be rewritten
$$
\begin{aligned}
\dot{\tilde q}(t)=&~\tilde p(t)\\
\dot{\tilde p}(t)=&~\nabla_{\tilde q} \log \pi(\tilde q).
\end{aligned}
$$

When the target distribution is approximately Gaussian with covariance matrix $\Sigma$, we can see that setting $M^{-1}=\Sigma$ standardizes the target distribution, which [Neal (2012)](#Neal) argues yields an optimal de-correlation.
On the other hand, if the target distribution is highly non-Gaussian, no one mass matrix can promise uniform energies.
To combat this, Riemann manifold HMC (RMHMC; [Girolami and Calderhead, 2011](#rmhmc)) argues for a functional mass matrix $M(q)$ to correct the target distribution based off of the position in parameter space.
The RMHMC approach is highly non-trivial, and so for the remainder of this post we consider HMC with position-invariant mass matrices.

There is one critical caveat to the aforementioned transformations.
Because large-scale matrix inversion is computationally costly, the practicality of the transformations is limited when the dimensionality of the target distribution is high.
As a result, one is often restricted to working with diagonal mass matrices.
Thus, the transformations only scale (and don't rotate) parameter and momentum space.
It's for this reason that mass matrices are commonly referred to in practice as scale matrices.
For the remainder of this post, we use the two naming conventions interchangeably.

### Variational inference
While common implementations of HMC take the elements of the scale matrix to be early estimates of marginal variances or precisions, variational inference (VI; [Blei et al., 2017](#VI)) can produce a black-box scale matrix that may be more optimal.
Given an intractable distribution $f,$ the goal of VI is to find the best approximation $g$ from some more tractable family $G$ by solving an optimization of the form
$$
g^*=\argmin_{g\in G}D(g,f),
$$
where $D$ is a divergence satisfying $D(g,f)\geq 0$ for all $g\in G$ and $D(g,f)=0$ if and only if $f=g$.
In most implementations of VI, $D$ is set as the Kullback-Leibler divergence $KL(g||f)$, which is defined
$$
KL(g||f)=\mathbb{E}_g\left[\log\frac{g}{f}\right].
$$

Applications of VI typically invoke a type of mean field assumption, choosing $G$ to be a family of factorized distributions even though $f$ itself doesn't factorize.
For example, for a target density $f$ that has a non-diagonal covariance matrix $M$, variational inference would approximate $f$ by some $g$ that has a diagonal covariance matrix $\Psi$.
This framework can be of natural use in initializing a scale matrix for HMC.

### Related work
This goal of using VI to initialize HMC is most related to the variationally inferred parameterization (VIP) algorithm proposed by [Gorinova et al. (2020)](#VIP), which automatically reparameterizes a Bayesian model into a probabilistic program whose posterior geometry is much more amenable to efficient MCMC.
While the authors use VI to search over a continuous space of reparameterizations, we suggest that a similar process can be done to search for an optimal scale matrix.

## Pre-conditioning HMC with VI
To reiterate, we want to leverage VI to approximate an ideal kinetic energy function—or equivalently, an ideal conditional momentum distribution—for high-dimensional models.
We consider a setting in which $f(p\mid q)$ is a zero-mean Gaussian with a non-diagonal mass matrix, and $g(p\mid q)$ is a zero-mean Gaussian with a diagonal mass matrix:
$$
\begin{aligned}
f(p\mid q)=&~\mathcal{N}(0,M)\\
g(p\mid q)=&~\mathcal{N}(0,\Psi),
\end{aligned}
$$
with $\Psi_{ij}=0$ whenever $i\neq j$.
<!-- Assuming our target distribution is Gaussian with covariance matrix $\Sigma$, so we effectively seek to approximate $M=\Sigma^{-1}$ by a diagonal matrix $\Psi$.  -->

Under this setting, [Margossian et al. (2024)](#VI-tradeoffs) prove an *impossibility theorem* that measures how well $\Psi$ can capture features of $M$.
The authors consider the following features:<br>
1. the marginal variances $M_{ii}$,
2. the marginal precisions $M_{ii}^{-1}$, and
3. the generalized variance $\det(M)$.
---
**Theorem 1 ([Margossian et al., 2024](#VI-tradeoffs))**<br>
Let $f$ and $g$ be distributions with covariances $M$ and $\Psi$, respectively, where $\Psi$ is diagonal but $M$ is not. Then
1. If $\text{diag}(\Psi)=\text{diag}(M)$, then $\det(\Psi)>\det(M)$ and $\text{diag}(\Psi^{-1})\leq\text{diag}(M^{-1})$, and this last inequality is strict for at least one of the diagonal elements.
2. If $\text{diag}(\Psi^{-1})=\text{diag}(M^{-1})$, then $\det(\Psi)<\det(M)$ and $\text{diag}(\Psi)\leq\text{diag}(M)$, and this last inequality is strict for at least one of the diagonal elements.
3. If $\det(\Psi)=\det(M)$, then $\Psi_{ii}<M_{ii}$ and $\Psi^{-1}_{jj}<M^{-1}_{jj}$ for at least one $i$ and $j$.
---
<span style="color:yellow">Why is this impossibility theorem useful?</span>

Recall that when the target distribution $f(q)$ is Gaussian with covariance matrix $\Sigma$, it's optimal to set $M=\Sigma^{-1}$.
Indeed, it's been shown that minimizing $KL(g||f)$ outputs a $g$ that matches marginal precisions ([Turner and Sahani, 2011](#Turner)).

This leaves us with two competing methods for initializing HMC when $f(q)$ is Gaussian.

1\. Run a VI scheme to initialize HMC.

$f(q)=N(\mu,\Sigma)$

$g(q)=N(\nu,\Psi^{-1})$

Then we obtain the priors $q\sim N(\nu,\Psi^{-1})$ and $(p\mid q)\sim N(0,\Psi)$. Set $\tilde p=\sqrt{\Psi^{-1}}p$ and $\tilde q=\sqrt{\Psi}(q-\nu)$.
Run HMC with the modifications $\tilde q\sim N(0,I)$ and $(\tilde p\mid \tilde q)\sim N(0,I)$.

2\. Do not use VI and instead run a burn-in period of HMC. 

Run HMC with priors $q\sim N(0,I)$ and $(p\mid q)\sim N(0,I)$ for some time $t^*$. 

$\nu_i=\frac{1}{t^*}\int_0^{t^*}q_i(u)du$
$$
\Psi_{ii}^{-1}=\frac{1}{t^*}\int_0^{t^*}q_i^2(u)du-\left(\frac{1}{t^*}\int_0^{t^*}q_i(u)du\right)^2.
$$




Exploit variance-precision tradeoff:

Introduce 


### Computational costs

## Simulation experiments




We minimize $KL(q||p)$ by (equivalently) maximizing the evidence lower bound (ELBO), and we estimate the ELBO via Monte Carlo with draws from $q$, a procedure at the core of modern implementations of “blackbox” VI (Kucukelbir et al., 2017).


We study two competing methods for initializing HMC for non-Gaussian targets.

1\. Reparameterize the target distribution. Run a VI scheme to initialize the scale matrix.

$f(p\mid q)=N(0,M)$

$g(p\mid q)=N(0,\Psi)$

Then we obtain the priors $q\sim N(\nu,\Psi^{-1})$ and $(p\mid q)\sim N(0,\Psi)$. Set $\tilde p=\sqrt{\Psi^{-1}}p$ and $\tilde q=\sqrt{\Psi}(q-\nu)$.
Run HMC with the modifications $\tilde q\sim N(0,I)$ and $(\tilde p\mid \tilde q)\sim N(0,I)$.

2\. Reparameterize the target distribution. Run a burn-in period of HMC to initialize the scale matrix. 

After reparameterizing, run HMC for some time $t^*$. 


$$
\Psi_{ii}^{-1}=\frac{1}{t^*}\int_0^{t^*}q_i^2(u)du-\left(\frac{1}{t^*}\int_0^{t^*}q_i(u)du\right)^2.
$$

#### Rosenbrock distribution

$$
\begin{aligned}
f(q_2\mid q_1)=&~N(0.03(q_1^2-100),1)\\
f(q_1)=&~N(0,10)
\end{aligned}
$$

$q=(q_1,q_2)$

#### Eight schools

$$
\begin{aligned}
f(y_i\mid \theta_i)=&~N(\theta_i,\sigma_i^2)\\
f(\theta_i\mid \mu,\tau)=&~N(\mu,\tau^2)\\
f(\tau)=&~N^+(0,10)\\
f(\mu)=&~N(5,3^2)
\end{aligned}
$$

$q=(\mu,\tau,\theta)$

$f(q\mid y,\sigma)$

#### German credit

$$
\begin{aligned}
f(y_i\mid \beta,x_i)=&~\text{Bernoulli}\left(\frac{1}{\exp(-\beta_{1:24}^Tx_i-\beta_0)}\right)\\
f(\beta)=&~N(0,I_{25})
\end{aligned}
$$

$q=\beta$

$f(q\mid y,x)$

#### Radon effect

$$
\begin{aligned}
f(\log y_i\mid \omega,\theta,\sigma,x_i)=&~N(\omega^Tx_i+\theta_{j[i]},\sigma^2)\\
f(\sigma)=&~\text{Uniform}(0,100)\\
f(\omega_k)=&~N(0,1)\\
f(\theta_j\mid \mu,\tau)=&~N(\mu,\tau^2)\\
f(\tau)=&~\text{Uniform}(0,100)\\
f(\mu)=&~N(0,1)
\end{aligned}
$$

$q=(\omega,\theta,\sigma,\mu,\tau)$

$f(q\mid y,x)$

#### Stochastic volatility

$$
\begin{aligned}
f(y_i\mid h_i)=&~N\left(0,\exp\left(\frac{h_i}{2}\right)\right)\\
h_{i>1}=&~\mu+\sigma z_i+\phi(h_{i-1}-\mu)\\
h_1=&~\mu+\frac{\sigma z_1}{\sqrt{1-\phi^2}}\\
f(z_i)=&~N(0,1)\\
f\left(\frac{\phi+1}{2}\right)=&~\text{Beta}(20,1.5)\\
f(\mu)=&~\text{Exp}(1)\\
f(\sigma)=&~\text{Cauchy}^+(0,2)
\end{aligned}
$$

$q=(z,\phi,\mu,\sigma)$

$f(q\mid y)$

## Discussion


## References
<span id="HMC">M. Betancourt. A conceptual introduction to Hamiltonian Monte Carlo. *arXiv:1701.02434v1*, 2018.

<span id="VI">D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational inference: a review for statisticians. *Journal of the American Statistical Association*, 112(518):859–877, 2017.

<span id="rmhmc">M. Girolami and B. Calderhead. Riemann manifold Langevin and Hamiltonian Monte Carlo. *Journal of the Royal Statistical Society: Series B (Statistical Methodology)*, 73:123-214, 2011.

<span id="VIP">M. I. Gorinova, D. Moore, and M. Hoffman. Automatic reparameterisation of probabilistic programs. In *Proceedings of the 37th International Conference on Machine Learning*. PMLR, 2020.

<span id="nouturn">M. D. Hoffman and Andrew Gelman. The no-U-turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. *Journal of Machine Learning Research*, 15:1593–1623, 2014.

<span id="variational-methods">M. I. Jordan, Z. Ghahramani, T. S. Jaakkola, and L. K. Saul. An introduction to variational methods for graphical models. *Machine Learning*, 37:183–233, 1999.

<span id="VI-tradeoffs">C. C. Margossian, L. Pillaud-Vivien, and L. K. Saul. Variational inference for uncertainty quantification: an analysis of trade-offs. *arXiv:2403.13748*, 2024.

<span id="Neal">R. M. Neal. MCMC using Hamiltonian dynamics. In *Handbook of Markov Chain Monte Carlo*. CRC Press, 2012.

<span id="Tran">J. H. Tran and T. S. Kleppe. Tuning diagonal scale matrices for HMC. *Statistics and Computing*, 34(196), 2024.

<span id="Turner">R. E. Turner and M. Sahani. Two problems with variational expectation maximisation for time-series models. In *Bayesian Time Series Models*. Cambridge University Press, 2011.

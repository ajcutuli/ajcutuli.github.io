---
layout: archive
permalink: /HMMVI/
header-includes:
    - \usepackage{bbm}
author_profile: true
title: "Resolving Trade-offs of Variational Inference for Hidden Markov Models via Normalizing Flows"
---

{% include base_path %}
### Abstract
Variational methods seek to provide bounds on probabilities by simplifying a graphical model in such a way that inference is efficient.
This is of immense interest in Bayesian statistics, where it's often the case that an intractable marginal likelihood needs to be approximated in lieu of exact sampling techniques that struggle to scale to high-dimensional models and large datasets.
There is a robust literature on variational inference for time-independent models but a relative lack of exploration into its efficacy for models with temporal dependecies.
This is partly due to the fact that a fully factorized approximation is not sensible for such models, since doing so ignores correlations that are integral to accurate inference.
Nonetheless, taking a structured approximation that assumes time dependencies is still practical with the help of batch inference, yet such an approach still fails to avoid pitfalls in uncertainty quantification that beset variational methods.
In this post, we study trade-offs in applying variational inference to hidden Markov models (HMMs), which are an important class of time-dependent models in a variety of fields, and study how normalizing flows can be leveraged to enhance accuracy.

## Introduction
As the demand for large-scale machine learning in stochastic and complex systems increases, so too does the demand for the accurate and scalable estimation of probabilistic models with temporal dependencies.
One critical class of such models is the hidden Markov model (HMM), which is used in a wide array of fields such as signal processing, finance, natural language, statistical mechanics, and biology.
As HMMs are typically concerned with studying stochastic processes, there is a heightened interest in describing model uncertainty to ensure that decisions based on model predictions are informed and accurate.
Hence, approaching the modeling task from a Bayesian perspective is often appropriate, since by taking a Bayesian approach, unknown model parameters can be treated as random variables, which allows for beliefs in the form of probability distibutions to update as information is observed.
However, designing an inference algorithm that scales to large time-dependent datasets is a challenge that requires making trade-offs in accurate uncertainty quantification.

A class of Bayesian inference algorithms that has achieved recognized success in scaling to large data sets is variational inference (VI; [Blei et al., 2017](#VI)).
By considering a tractable family of distributions $Q$, VI approximates the intractable posterior $p$ by minimizing a divergence measure between $p$ and some $q\in Q$.

Most applications of VI invoke a type of mean field assumption, choosing $Q$ to be a family of factorized distributions even though $p$ itself doesn't factorize.
As a result, this $Q$ may not be sufficiently rich to capture all the features of $p$, leading to a compromised $q$ that neglects dependencies in the true posterior.
<span style="color:yellow">[example: marginal and generalized variance]</span>
Recently, ([Margossian et al., 2024](#VI-tradeoffs)) studied this problem in the setting of a Gaussian $p$ with a dense covariance matrix approximated by a Gaussian $q$ with a diagonal covariance matrix.

By definition, a fully factorized approach is not sensible for time series models because of temporal dependencies that are assumed between some variables.
However, most time series models are adorned with parameters that are not time-dependent but may exhibit alternative correlation structures that are lost by factorizing them.
In this post, we study the trade-offs of factorized VI in models of time-dependent nature, through an exploration of HMMs.
<!-- Despite offering asymptotic exactness, sampling algorithms for posterior inference often struggle to scale to the size of modern models, and although Hamiltonian Monte Carlo has proven to be an effective way to remedy the inscalability of MCMC in many cases, these methods require careful parameter tuning that make them difficult to use in practice. -->
<!-- Hence, optimization-based approaches such as variational inference have been widely adopted to quickly approximate high-dimensional Bayesian posteriors. -->


## Background
### Variational inference
Given an intractable distribution $p$, the problem of variational inference (VI) is to find the best approximation $q$ from some more tractable family $Q$ by solving an optimization of the form
$$
q^*=\argmin_{q\in Q}D(q,p),
$$
where D is a divergence satisfying $D(q,p)\geq 0$ for all $q\in Q$ and $D(q,p)=0$ if and only if $p=q$.

### Normalizing flows
<span style="color:red">Normalizing flows operate by pushing a simple density through a series of transformations to produce a richer, potentially more multi-modal distribution—like a fluid flowing through a set of tubes.
The main idea of flow-based modeling is to express the variables of interest $\mathbf{x}\in\mathbb{R}^N$ as a transformation $T$ of a vector $\mathbf{u}\in\mathbb{R}^N$ sampled from $p_u(\mathbf{u})$:</span>
$$\mathbf{x}=T(\mathbf{u}),$$
<span style="color:red">where $\mathbf{u}\sim p_u(\mathbf{u})$.
The defining property of flow-based models is that the transformation $T$ must be a diffeomorphism: $T$ must be invertible and both $T$ and $T^{-1}$ must be differentiable.</span>
<span style="color:red">Under these conditions, the density of x is well-defined and can be obtained by a change of variables utilizing the Jacobian of $T$:</span>
$$p_x(\mathbf{x})=p_u(\mathbf{u})~|\det J_T(\mathbf{u})|^{-1}.$$
<span style="color:red">In practice, we often construct a flow-based model by implementing $T$ or $T^{-1}$ with a neural network and taking $p_u(\mathbf{u})$ to be a simple density such as a multivariate normal.
An important property of invertible and differentiable transformations is that they are composable. Given two such transformations $T_1$ and $T_2$, their composition $T_2 \circ T1$ is also invertible and differentiable.
In consequence, we can build complex transformations by composing multiple instances of simpler transformations, without compromising the requirements of invertibility and differentiability, and hence without losing the ability to calculate the density $p_x(\mathbf{x})$.</span>


### Hidden Markov models
Hidden Markov models (HMMs) are a class of stochastic processes consisting of observations $\mathbf{y}=(y_1,\dots,y_T)$ and latent states $\mathbf{x}=(x_1,\dots,x_T)$ generated by a Markov chain of $K$ states.
A HMM observed at stationarity is defined by a $K\times K$ transition matrix $A$ in which $A_{ij}=\mathbb{P}(x_t=j\mid x_{t−1}=i)$ and emission parameters $\phi=\{\phi_k\}_{k=1}^K$ describing the conditional probability of $y_t$ given $x_t$.
The initial state $x_1$ is sampled from the leading left-eigenvector of $A$, which gives the stationary transition probability $\pi$.
Of course, one can alternatively assume the sequence is not observed at stationarity, which would require the modeling of an initial state distribution $\pi_0$.
The joint likelihood factorizes as
$$
\begin{aligned}
p(\mathbf{x},\mathbf{y}\mid A,\phi)=&~p(\mathbf{y}\mid \mathbf{x},\phi)p(\mathbf{x}\mid A)\\
=&~p(x_1\mid \pi)p(y_1\mid x_1,\phi)\prod_{t=2}^Tp(x_t\mid x_{t-1},A)p(y_t\mid x_t,\phi).
\end{aligned}
$$
We abuse notation slightly here in writing $p(x_1\mid\pi)$, since $\pi$ may be determined from $A$.

As in ([Foti et al., 2014](#SVI-HMM)), we focus on a class of Bayesian HMMs in which transition probabilities are given conjugate Dirichlet priors, emission parameters are given conjugate normal-inverse-Wishart (NIW) priors, and the conditional probability of $y_t$ given $x_t$ is given a multivariate Gaussian prior:
<span style="color:yellow">[need to encode dependencies in the priors]</span>
$$A_i\sim\text{Dir}(\alpha_i)~~~~~~~~~x_1\sim\pi~~~~~~~~~x_{t}\mid x_{t-1}\sim A_{x_{t-1}}~~~~~~~~$$
$$
\phi_k = (\mu_k, \Sigma_k)\sim \text{NIW}(\mu_0, \kappa_0, \Sigma_0, \nu_0)~~~~~~~~~~~~~~y_t\mid x_t\sim \text{N}(\mu_{x_t}, \Sigma_{x_t}).$$
<!-- In this context, Bayes formula comes out to 
$$
p(\pi,A,\mathbf{x},\phi\mid\mathbf{y})=\frac{p(\mathbf{x},\mathbf{y}\mid\pi,A,\phi)p(\pi,A,\phi)}{Z}.
$$ -->

#### Structured mean field approximation for HMMs
We are interested in studying the problem of inferring the posterior distribution of the hidden state sequence and model parameters given an observation sequence, denoted $p(\mathbf{x},A,\phi\mid\mathbf{y})$.
We consider a structured mean field approximation that is standard for HMMs:
$$q(\mathbf{x},A,\phi)=q(\mathbf{x})q(A)q(\phi).$$
Notice that since $\mathbf{x}$ is a sequence of time-dependent variables, factorizing any further leaves out information that is obviously critical.
However, by its very nature, the factorized approximation cannot estimate correlations encoded by the priors on the hidden state sequence and model parameters.

<!-- The joint likelihood of the HMM factorizes as
$$
\begin{aligned}
p(\mathbf{x},\mathbf{y}\mid \pi,A,B)=&~p(\mathbb{y}\mid x,B)p(\mathbb{x}\mid A,\pi)\\
=&~\left[\prod_{t=1}^Tp(y_t\mid x_t,B)\right]\left[p(x_1\mid \pi)\prod_{t=2}^Tp(x_t\mid x_{t-1},A)\right]\\
=&~\left[\prod_{t=1}^T \prod_{k=1}^K \prod_{v=1}^V B_{kv}^{\mathbb{1}(x_t=k,y_t=v)}\right]\left[\prod_{k=1}^K \pi_k^{\mathbb{1}(x_1=k)}\right]\left[\prod_{t=2}^T \prod_{i=1}^K \prod_{j=1}^K A_{ij}^{\mathbb{1}(x_{t-1}=i,x_t=j)}\right]
\end{aligned}
$$ -->

<!-- The joint distribution factorizes as
$$
p(\mathbf{x},\mathbf{y})=
\pi(x_1)p(y_1\mid x_1)\prod_{t=2}^{T}p(x_t\mid x_{t-1},A)p(y_t\mid x_t,\phi)
$$
where $A=\left[A_{ij}\right]_{i,j=1}^K$ is the transition matrix with $A_{ij} = \mathbb{P}(x_t=j\mid x_{t−1}=i), \phi = \{\phi_k\}_{k=1}^K$ are the emission parameters, and $\pi_0$ is the initial distribution. -->


## References
<span id="Beale">M. J. Beale. *Variational Algorithms for Approximate Bayesian Inference*. Ph.D. thesis, University College London, 2003.

<span id="SVI-HMM">N. J. Foti, J. Xu, D. Laird, and E. B. Fox. Stochastic variational inference for hidden Markov models. In *Advances in Neural Information Processing Systems 27*, 2014.

<span id="VI-tradeoffs">C. C. Margossian, L. Pillaud-Vivien, and L. K. Saul. Variational inference for uncertainty quantification: an analysis of trade-offs. *arXiv:2403.13748*, 2024.

<span id="variational-methods">M. I. Jordan, Z. Ghahramani, T. S. Jaakkola, and L. K. Saul. An introduction to variational methods for graphical models. *Machine Learning*, 37:183–233, 1999.

<span id="VI">D. M. Blei, A. Kucukelbir, and J. D. McAuliffe. Variational inference: a review for statisticians. *Journal of the American Statistical Association*, 112(518):859–877, 2017.

<span id="normalizing-flows">G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan. Normalizing flows for probabilistic modeling and inference. *Journal of Machine Learning Research*, 22(1):1-64, 2021.

Stochastic Variational Inference for Bayesian Time Series Models

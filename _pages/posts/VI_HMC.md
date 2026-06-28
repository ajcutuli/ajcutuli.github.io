---
layout: archive
permalink: /VI_HMC/
author_profile: true
title: "Variational Approach to Pre-Conditioning Hamiltonian Monte Carlo"
---

{% include base_path %}

## Mathematical Background

<span style="color:yellow">What the mass matrix does geometrically</span>

Given momentum $p\sim\mathcal{N}(0,M)$, we have 
$$
H(q,p)=U(q)+\frac{1}{2}p^\intercal M^{-1}p,
$$
where $U(q)=-\log\pi(q)$. 
For a small neighborhood where $U$ is approximately quadratic (i.e. $\pi$ is locally Gaussian),
$$
U(q)\approx U(q^*) + \frac{1}{2}(q-q^*)^\intercal \mathbf{H}^*(q-q*),
$$
where $\mathbf{H}^*:=\nabla^2 U(q^*)\succ0$.
Then Hamilton's equations are
$$
\dot{q}=M^{-1}p,~~~~\dot{p}=-\mathbf{H}^*(q-q^*).
$$
Differentiating gives 
$$
\ddot{q}=-M^{-1}\mathbf{H}^*(q-q^*).
$$
Now, since $M^{-1}\mathbf{H}^*$ is symmetric positive definite, we can diagonalize to obtain
$$
M^{-1}\mathbf{H}^*=V\Lambda V^\intercal,
$$
where the columns of $V$ give normalized eigenvectors and $\Lambda=diag(\lambda_1,\dots,\lambda_d)$ gives eigenvalues.
Under the rotated coordinate system $y = V^\intercal q$, we have 
$$
\ddot y = V^\intercal \ddot q = -V^\intercal M^{-1}\mathbf{H}^*q=-V^\intercal (V\Lambda V^\intercal) q=-\Lambda V^\intercal q=-\Lambda y
$$
and thus $\ddot y_i = -\lambda_i y_i$ for $i=1,\dots,d$.
This ODE bears the general solution $y_i(t)=a\cos(\sqrt{\lambda_i}t)+b\sin(\sqrt{\lambda_i}t)$. Rotating back the coordinate system, we conclude $q(t)$ is the sum of harmonic oscillations in each eigenvector direction of $M^{-1}\mathbf{H}^*$ whose squared frequencies are the eigenvalues of $M^{-1}\mathbf{H}^*$.


Call $\Omega^2=M^{-1}\mathbf{H}^*$. Thus, the system decouples into harmonic oscillators with frequencies $\omega_i=\sqrt{\lambda_i(\Omega^2)}$.

The spread of these frequencies (i.e. the condition number $\kappa(\Omega^2)=\frac{\lambda_{max}}{\lambda_{min}}$) governs stiffness, admissible leapfrog step size $\varepsilon$, and acceptance.

We want to choose $M\succ0$ to minimize stiffness, so naturally we have the objective
$$
\min_{M\succ0} \kappa(M^{-1}\mathbf{H}^*).
$$


## The Idea

This replaces heuristic identity/diagonal starts with a theoretically grounded, curvature-aware initializer that (i) reduces the spectral spread of the effective Hessian, (ii) permits larger stable \epsilon, and (iii) improves early-phase mixing and adaptation.


VI provides a data-driven Gaussian approximation $q_\phi$. Setting $M\approx\Sigma_\phi^{-1}$ is mathematically optimal for Gaussian targets and optimal in an averaged spectral sense for general targets, directly minimizing the geometric obstacles (condition number, stiffness) that limit HMC efficiency.


## Optimality in the Gaussian case



## Chasing optimality for general targets

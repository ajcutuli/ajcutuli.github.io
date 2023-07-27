---
layout: archive
permalink: /blog/Hawkes/
author_profile: true
title: "Hawkes Processes and Time Clustering in Finance"
---

{% include base_path %}

## Introduction

On May 6, 2010, U.S. equity markets experienced a then unprecedented intraday price drop and recovery in what would come to be known as the 2010 Flash Crash. Within the five minutes following 2:32 pm EDT, a multi-trillion dollar pullback left the Dow Jones Industrial Average nearly 10\% below its opening price for the day. The swift and dramatic drop triggered a half hour of such intense trading volume and market recovery that a 2014 CFTC report would call it “one of the most turbulent periods” in the history of U.S. financial markets[<sub>[1]</sub>](#ref1). On October 15, 2014, the U.S. Treasury market experienced a similarly drastic excitement[<sub>[2]</sub>](#ref2), and on October 7, 2016, the sterling took a 9\% dive against the dollar in Asian markets before recovering most of the loss within minutes[<sub>[3]</sub>](#ref3). These flash events are a byproduct of modernizing electronic markets, and few traditional models of financial processes account for such acute feedback effects. In recent years, however, the Hawkes process has been used to trace these observed phenomena.

$$
\begin{align}
    \lambda (t) = \lambda_0 (t) + \int_{-\infty}^{t} v(t-s) \text{d}N(s)
\end{align}
$$

Classically utilized for the modeling of earthquakes and their aftershocks[<sub>[4]</sub>](#ref4), Hawkes processes are a class of Poisson point processes whose rate is characterized by an exogenous component capturing a latent arrival rate and an endogenous component capturing the realized rate’s dependency on past arrivals, which are recorded in the counting process $N(s)$. In particular, through the exogenous term $\lambda_0 (t)$, any non-stationarities in the underlying process can be accounted for, and the Hawkes kernel $v$ can be fit to explain the feedback effects that past arrivals have on the present rate. Hence, Hawkes processes offer flexible and interpretable characterizations of many complex phenomena in finance, including, but not limited to, the temporal clustering of trades, mid-price volatility, and credit default events[<sub>[5,</sub>]](#ref5)[<sub>[6,</sub>]](#ref6). In this article, we outline calibration techniques for Hawkes processes and conclude with practical takeaways.

## Empirical Calibration
Empirically calibrating the Hawkes process in financial applications consists of pre-processing intra-day data to capture seasonality and then estimating the Hawkes kernel via maximum likelihood estimation. Here, we paraphrase much of section 9.3 of [<sub>[7]</sub>](#ref7) but outline derivation steps that are omitted from the text.

### Capture Intra-day Seasonality
Calendar time can be re-parameterized as business time by measuring the the intra-day activity pattern $\bar \lambda(t)$ as the average activity at time $t$ over many days and subsequently defining

$$
\begin{align*}
    \hat t(t) = \int_0^t \bar \lambda(s)\text{d}s.
\end{align*}
$$

Doing so allows for the re-parameterized activity to appear stationary i.e. $\lambda_0(t) = \lambda_0$.

### Estimate the Hawkes Kernel
Given the process measured in business time, the Hawkes kernel can be estimated using traditional statistical methods such as maximum likelihood estimation, as mentioned earlier.

#### Maximum Likelihood Estimation
Assuming that the kernel function $v(t)$ can be expressed in a parametric form with a parameter vector $\pmb{\theta}$, maximum likelihood estimation (MLE) can employed to estimate $\pmb{\theta}$ based on an observed realization of the point process. For $N$ events ocurring in $[0,T)$, the interval can be discretized into subintervals so that the likelihood function can be written as 

$$
\begin{align*}
    \mathbf{L}(t_1,t_2,...,t_N|\pmb{\theta})
    &= \left( \prod_{i=1}^N \lambda(t_i|\pmb{\theta}) e^{-\int_{t_{i-1}}^{t_i} \lambda(s|\pmb{\theta})\text{d}s}\right) e^{-\int_{t_{N}}^{T} \lambda(s|\pmb{\theta})\text{d}s}\\
    &= \left( \prod_{i=1}^N \lambda(t_i|\pmb{\theta})\right) e^{-\int_0^T \lambda(s|\pmb{\theta})\text{d}u} 
\end{align*}
$$

and the log-likelihood as

$$
\begin{align}
    \log \mathbf{L}(t_1,t_2,...,t_N|\pmb{\theta}) 
    = \sum_{i=1}^N \log \lambda(t_i|\pmb{\theta}) - \int_0^T \lambda(s|\pmb{\theta})\text{d}s.
\end{align}
$$

Here, 

$$
\begin{align*}
    \lambda(t_i|\pmb{\theta}) e^{-\int_{t_{i-1}}^{t_i} \lambda(s|\pmb{\theta})\text{d}s} \text{d}t
\end{align*}
$$

is the probability of an arrival coming in $[t_i,t_i+\text{d}t)$ and not in $(t_{i-1},t_i)$, which is a definition of inhomogeneous Poisson processes. It is important here to note that while a closed-form solution to $\pmb{\theta}$ would offer nice statistical properties, it is often not attainable in practice unless we assume that the kernel function has a certain parametric form. If this assumption is omitted, it is best to estimate (1) via numerical optimization or algorithmically[<sub>[8]</sub>](#ref8). However, imposing, for example, that $v$ is of exponential form (i.e. $v(t) = gwe^{-wt}$) allows for the direct computation of the maximum likelihood estimator for $\pmb{\theta}$.

With this exponential kernel, the parameter vector is $\pmb{\theta} = (\lambda_0,w,g)$, where $w^{-1}$ determines the time at which events stop exciting the process and $g=\int_0^{\infty} v(s) \text{d}s$ is the norm or branching ratio i.e. the expected number of child events that a parent event produces. A larger $g$ implies that a large fraction of events come in response to past events i.e. the activity is highly endogeneous. The stochastic differential equation governing the process can be derived by taking the differential on either side of (1):

$$
\begin{align*}
    \text{d}\lambda(t)
    &=\text{d}\lambda_0 + \text{d}\int_{-\infty}^t v(t-s)\text{d}N(s)\\
    &=\left(\int_{-\infty}^t \frac{\text{d}v(t-s)}{\text{d}t}\text{d}N(s)\right)\text{d}t + v(0)\text{d}N(t)\\
    &=-w\left(\int_{-\infty}^tv(t-s)\text{d}N(s)\right)\text{d}t + gw\text{d}N(t)\\
    &=-w(\lambda(t)-\lambda_0)\text{d}t + gw\text{d}N(t).
\end{align*}
$$

This essentially describes the intensity $\lambda(t)$ as decaying exponentially toward $\lambda_0$ but being excited by an amount $gw$ when an event occurs.

The exponential Hawkes is Markovian in nature, and, given arrival times $t_1,t_2,...,t_N$ in $(0,T]$, this enables (1) to be discretized to

$$
\begin{align*}
    \lambda(t_i) 
    &= \lambda_0 + \sum_{k=1}^{i-1} v(t_i-t_k) \\
    &= \lambda_0 + gw\sum_{k=1}^{i-1} e^{-w(t_i-t_k)}.
\end{align*}
$$

Letting $Z_i=\sum_{k=1}^{i-1} e^{-w(t_i-t_k)}$, (2) can be simplified to

$$
\begin{align*}
    \log \mathbf{L}(t_1,t_2,...,t_N|\pmb{\theta}) 
    &= \sum_{i=1}^N \log \lambda(t_i|\pmb{\theta}) - \int_0^T \lambda(s|\pmb{\theta})\text{d}s \\
    &= \sum_{i=1}^N \log (\lambda_0 + gwZ_i) - \int_0^T \left(\lambda_0 + gw \sum_{k=1}^{N(s)-1} e^{-w(t_{N(s)}-t_k)}\right)\text{d}t_{N(s)} \\
    &= \sum_{i=1}^N \log (\lambda_0 + gwZ_i) - \lambda_0T -\int_0^T gw \sum_{k=1}^{N(s)-1} e^{-w(t_{N(s)}-t_k)}\text{d}t_{N(s)} \\
    &= \sum_{i=1}^N \log (\lambda_0 + gwZ_i) - \lambda_0T - \sum_{i=1}^{N} g\int_{t_i}^T we^{-w(t_{N(s)}-t_i)}\text{d}t_{N(s)}\\
    &= \sum_{i=1}^N \log (\lambda_0 + gwZ_i) - \lambda_0T - \sum_{i=1}^{N} g(e^{-w(T-t_i)}-1) \\
    &= - \lambda_0T + \sum_{i=1}^N \left[\log (\lambda_0 + gwZ_i) - g(e^{-w(T-t_i)}-1)\right].
\end{align*}
$$

Evaluating this expression computationally is an $O(N^2)$ operation, but $Z_i$ can be written as a recursion such that the problem is reduced to $O(N)$. Namely, recognizing

$$
\begin{align*}
    Z_i = e^{-w(t_i-t_{i-1})}\sum_{k=1}^{i-1} e^{-w(t_{i-1}-t_k)} = (1+Z_{i-1})e^{-w(t_i-t_{i-1})}
\end{align*}
$$

allows for the log likelihood and its gradients to be efficiently computed in order to find the MLE for $\pmb{\theta}$.

Another popular kernel function is the power-law kernel, which can be approximated as a sum of finite exponential kernels[<sub>[9]</sub>](#ref9), and we omit discussing its MLE procedure for brevity. Empirically, the power-law kernel is the more often observed feedback structure[<sub>[10,</sub>]](#ref10)[<sub>11\]</sub>](#ref11).

While MLE is an effective tool, the estimator can be quite biased in small samples and very sensitive to the choice of kernel function over short lags. An alternative method for calibrating the Hawkes process is the method of moments, which consists of matching theoretical and empirically observed first- and second-order moments in order to estimate $\pmb{\theta}$. Goodness-of-fit testing can be used to compare the estimated intensity $\hat \lambda(t)$ to the data under the null hypothesis that the data is generated by a Hawkes process, implying the residual process $\epsilon(t_i) = \int_0^{t_i} \hat \lambda(t) \text{d}t$ is Poisson with unit intensity[<sub>[12]</sub>](#ref12).

## Practical Takeaways

Hawkes processes can be a powerful way to parametrically address the time clustering of events and intermittent arrivals exhibited by many complex phenomena in finance. The key component of a Hawkes model that differentiates it from the more basic Poissonian approach is the self-excitation term that seeks to capture feedback effects from past arrivals.

A brief list of Hawkes representations used in finance includes models of market volatility\cite{filiminov-reflexivity}, price impact\cite{price-impact}, and order books\cite{Toke2011}. In \cite{cartea-hft}, a high frequency trading strategy is developed that relies on market orders arriving according to a Hawkes process, and \cite{Hawkes-Credit} use the Hawkes framework to describe portfolio loss due to credit default events.

In practice, such phenomena exhibit high endogeneity (i.e. $g$ close to 1), suggesting that current markets operate in a close to unstable system (hence the propensity for flash crashes) and/or that Hawkes processes are too simple a framework to model financial markets. The linear Hawkes process that we discussed have been extended by introducing covariates such as price returns\cite{cojumps-hawkes-factor} and news\cite{Rambaldi2014ModelingFE} to improve the ability the explain phenomena. As a financial modeler, it is critical to constantly bear in mind that, as put best in Derman and Wilmott's manifesto\cite{manifesto}, "models are not the world."

## References
1. <span id="ref1">A. Kirilenko, A. S. Kyle, M. Samadi, and T. Tuzun. The flash crash: High-frequency trading in an electronic market. *The Journal of Finance*, 72(3):967–998, 2017.<br>
2. <span id="ref2">M. J. Fleming, P. Johansson, F. M. Keane, and J. Meyer. From the vault: A look back at the october 15, 2014, flash rally. *Federal Reserve Bank of New York*, 2019.<br>
3. <span id="ref3">R. Jackson, L. Crowley-Reidy, and A. Schrimpf. The sterling ‘flash event’ of 7 october 2016. *Markets Committee Papers*, (9):1–29, 2017.<br>
4. <span id="ref4">Y. Ogata. Statistical models for earthquake occurrences and residual analysis for point processes. *Journal of the American Statistical Association*, 83(401):9–27, 1988.<br>
5. <span id="ref5">C. Bowsher. Modelling security market events in continuous time: Intensity based, multivariate point process models. *Journal of Econometrics*, 141(2):876–912, 2007.<br>
6. <span id="ref6">E. Errais, K. Giesecke, and L. R. Goldberg. Affine point processes and portfolio credit risk. *SIAM Journal on Financial Mathematics*, 1(1):642–665, 2010.<br>
7. <span id="ref7">J. Bouchaud, J. Bonart, J. Donier, and M. Gould. *Trades, Quotes and Prices: Financial Markets Under the Microscope*. Cambridge University Press, 2018.<br>
8. <span id="ref8">J. Olson and K. Carley. Exact and approximate em estimation of mutually exciting hawkes processes. *Statistical Inference for Stochastic Processes*, 16:63–80, 2013<br>
9. <span id="ref9">T. Bochud and D. Challet. Optimal approximations of power laws with exponentials: application to volatility models with long memory. *Quantitative Finance*, 7(6):585–589, 2007.<br>
10. <span id="ref10">S. J. Hardiman, N. Bercot, and J. Bouchaud. Critical reflexivity in financial markets: a hawkes process analysis. *The European Physical Journal B*, 86(10):442, 2013.<br>
11. <span id="ref11">E. Bacry and J.F. Muzy. First- and second-order statistics characterization of hawkes processes and non-parametric estimation. *IEEE Transactions on Information Theory*, 62(4):2184–2202, 2016.<br>
12. <span id="ref12">F. Papangelou. Integrability of expected increments of point processes and a related random change of scale. *Transactions of the American Mathematical Society*, 165:483–506, 1972.<br>

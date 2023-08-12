---
layout: archive
permalink: /blog/CNN-LSTM-LOB/
author_profile: true
title: "Trading in the Limit Order Book with CNN-LSTM"
---

{% include base_path %}

**Note:** This is largely a replication of work done by the likes of Petter Kolm, Nicholas Westray, Jeremy Turiel, Zihao Zhang, Stefan Zohren, and others. I don't claim that this is novel work but instead that I learned a lot. On this page, I give an abridged version of the Jupyter notebook available [here](https://github.com/ajcutuli/OFI_NN_Project/blob/main/DeepOFI.ipynb).

## Abstract
In the notebook, we discuss and implement an artificial neural network originally employed by Zhang et al[<sub>[1]</sub>](#ref1) that combines convolutional neural networks (CNNs) and a long short-term memory (LSTM) neural network in order to classify future directions of an order book at a high frequency. Specifically, given Coinbase order book data for Bitcoin, we seek to predict whether the mid price increases, decreases, or does not change in the next observation of the time series. Unlike Zhang et al's papers we reference[<sub>[1]</sub>](#ref1)[<sub>[2]</sub>](#ref2), which use non-stationary order book states as inputs to the network, our instantiation of the architecture is trained on order flow and order flow imbalance, which are stationary quantities derived from the limit order book[<sub>[3]</sub>](#ref3). Hence, this discussion also draws heavy inspiration from a 2021 article by Kolm et al[<sub>[4]</sub>](#ref4), which demonstrated that forecasting using order flow significantly outperforms raw order book inputs. We further this by doing an analysis of the impact that differencing order flow into order flow imbalance has on the forecasting performance of the model. We also do a deeper dive on the autoregressive structures of the data than the authors do. After the training procedure, we make an attempt at a intra-second trading strategy using the models.

## Order Books, Flow, and Imbalance
Today's trading of equities and other securities is often facilitated by a limit order book, which collects bids and offers made by prospective buyers and sellers and determines which incoming orders get executed and which are added to the book. The bid price is the highest price buyers are prepared to buy at, and the ask price is the lowest price sellers are willing to sell at. The mid price, which our model seeks to predict moves in, is the midpoint of the bid price and the ask price.

An order is defined by its side, quantity demanded, price to trade at, and time of submission. As one enters the system, the matching engine of the exchange tries to match the order with existing orders in the book. Orders that match are executed and called market orders, and orders that do not match or only partially match are added to the book and called limit orders.

We pass into our model representations of the first ten levels of the order book. Each observation in our dataset will be a 40-variable vector displaying the price and volume for each of the top ten bids and asks, giving us a truncated screenshot of the *state of the limit order book* at each timestep.

$$ \begin{equation*} \text{s}_t^{LOB} := (a_t^1, v_t^{1,a}, b_t^1, v_t^{1,b}, ..., a_t^{10}, v_t^{10,a}, b_t^{10}, v_t^{10,b})^T \in \mathbb{R}^{40} \end{equation*} $$

We define the *bid order flows* (bOF) and *ask order flows* (aOF) at a timestamp to be 10-variable vectors computed using two consecutive order book states, where each element is given by

$$ \text{bOF}_{t,i} :=   \left\{
\begin{array}{ll}
      v_t^{i,b}, & b_t^i > b_{t-1}^i \\
      v_t^{i,b} - v_{t-1}^{i,b}, & b_t^i = b_{t-1}^i \\
      -v_t^{i,b}, & b_t^i < b_{t-1}^i \\
\end{array} 
\right. $$

$$ \text{aOF}_{t,i} :=   \left\{
\begin{array}{ll}
      -v_t^{i,a}, & a_t^i > a_{t-1}^i \\
      v_t^{i,a} - v_{t-1}^{i,a}, & a_t^i = a_{t-1}^i \\
      v_t^{i,a}, & a_t^i < a_{t-1}^i \\
\end{array} 
\right. $$

for $i = 1, ..., 10$. With this, we define *order flow* (OF)

$$ \begin{equation*} \text{OF}_t :=  (\text{bOF}_{t,1}, \text{aOF}_{t,1}, ..., \text{bOF}_{t,10}, \text{aOF}_{t,10})^T \in \mathbb{R}^{20} \end{equation*} $$

and *order flow imbalance* (OFI)

$$ \begin{equation*} \text{OFI}_t := \text{bOF}_t - \text{aOF}_t \in \mathbb{R}^{10} \end{equation*} . $$

While a sequence of limit order book states is a complex non-stationary process, the above formulas for order flow and order flow imbalance transform consecutive order book states into a stationary process. This property allows for our eventual test test of the deep learning model to be reasonably similar to the training set and thus appropriate to predict off of using the model. It also allows for more ease in the learning of long-term dependencies by our LSTM layer, which Kolm et al see as a reason behind their finding that sequence length only marginally impacted model performance[<sub>[4]</sub>](#ref4). On a separate note, when trained on order flow, which keeps the bid and ask sides separate, the CNN layers of our model will be given the added flexibility of being able to combine bid and ask order flows asymmetrically, so we expect that our forecasting model will perform better on order flow than on order flow imbalance.

## Introducing the CNN-LSTM Model
Different neural network architectures can compress data and change their behavior over time in a way that supports their efficacy in difficult modeling situations, which is a quality particularly useful for the learning of complex financial time series. Although architectures can be used individually, they are often complementary in their modeling capabilities and, when used together, can learn unique structures in data and improve a model's ability to execute a desired task. For instance, our CNN-LSTM model architecture we adopt consists of CNN layers and an Inception Module that compress and wrap the order book data in a manner that allows a LSTM module to learn temporal dependencies from a smaller parameter space, leading to a more parsimonious model[<sub>[1]</sub>](#ref1).

*Convolutional neural network* (CNN) layers reduce the dimension of the multivariate input by aggregating bid and ask sides and levels in the order book. The output of these convolutional layers serve as an input to the *Inception Module*, which wraps convolutions together to capture behavior over multiple timescales, serving as a sort of moving average indicator whose decay weights are learned via backpropagation[<sub>[1]</sub>](#ref1). Outputs of the Inception Module are concatenated and reshaped into an input to the *long short-term memory* (LSTM) layer.

As in Zhang et al[<sub>[2]</sub>](#ref2), we apply variational dropout as a stochastic regularizer to reduce overfitting and make decisions that take into account the epistemic uncertainty of the estimted parameters. In particular, random sampling of predictions is done with *Monte-Carlo (MC) dropout* to generate multiple predictions, which we average in order to––in theory––improve out-of-sample performance.

Lastly, since we formulate this forecasting problem as one of classification, we add an output layer with a softmax activation function, resulting in a final output whose elements represent the estimated probability of observing each price movement in the next timestamp.

## Data
Our data-scraping method pulls live order book state information for Bitcoin traded on Coinbase. The data we obtain is not nearly as granular as the expensive datasets found on LOBSTER, so we only try to forecast relative changes in the next event instead of multiple horizons.

## Methodology and Experimentation
We adopt a Box-Jenkins approach to modeling the time series by first recognizing the vector autoregressive (VAR) nature of the process to infer how long our LSTM lookback window should be. We then cross-validate hyperparameters and subsequently train the model.

### VAR Order Selection
To select optimal lags, we iteratively fit order flow and imbalance data as VAR models with an increasing estimate for the order $p$ and selecting the estimate $\hat{p}$ that minimizes Akaike information criteria (AIC). This, however, does not guarantee that the deep learning model does not underfit the data.

### Dropout Tuning with Cross-Validated Grid-Search
After properly scaling and tensorizing our data, we apply time series cross-validated grid-search to assess the bias-variance tradeoff of selecting different hyperparameters. As advocated for in BDLOB[<sub>[2]</sub>](#ref2), we employ variational dropout as a stochastic regularizer in the deep neural network and exhaustively compare scoring over several different dropout rates. We apply early stopping to avoid overfitting by terminating training when validation loss has not improved for 5 consecutive epochs[<sub>[4]</sub>](#ref4).

### Evaluation
After training the models, we use MC dropout to better approximate the conditional expectations at each out-of-sample observation by performing 100 forward passes on each model and storing the average. That is, as the number of out-of-sample predictions we execute grows very large, the average across those predictions will approach the true conditional probability of an outcome given the training data.

## Building a Trading Signal
We compare two trading strategies using our test set as the trading period. We ignore transaction costs and close positions at the end of the trading period. For each timestamp we trade at, we go long or short $\mu$ Bitcoins, where $\mu$ is 30% of the volume at the first ask or bid level we enter at, respectively[<sub>[2]</sub>](#ref2).

### Softmax Trading Strategy
For this strategy, we choose a threshold probability $\alpha$ and go long if 
$ \hat{p}_{1,t}>\alpha $ and go short if 
$ \hat{p}_{-1,t}>\alpha $, where 
$ \hat{p}_{1,t} $ is the predicted probability of an upward move at time 
$ t $ and 
$ \hat{p}_{-1,t} $ is the predicted probability of a downward move at time 
$ t $. Only one position is allowed at any time. We store the cumulative profits and their ratios to transaction volume for a few threshold values.

### Bayesian Trading Strategy
To leverage uncertainty information, we use *predictive entropy* to summarize classification uncertainty due to variational dropout[<sub>[2]</sub>](#ref2). The metric of predictive entropy $\mathbb{H}$ follows from our aforementioned understanding of conditional expectation by making use of the predictive distribution captured by our 100 forward passes from earlier. For an input $ x_t$, a predicted output $y_t$, training data $\mathcal{D}_{\text{train}}$, and estimated model parameters $\hat{w}$, we define predictive entropy by
$$
\begin{equation*}
     \begin{aligned}
     \mathbb{H}(y_t|x_t,\mathcal{D}_{\text{train}}) &= -\sum_{j=-1}^1 p(y_t=j|x_t,\mathcal{D}_{\text{train}})\log p(y_t=j|x_t,\mathcal{D}_{\text{train}}) \\
    &\approx -\sum_{j=-1}^{1} \left( \frac{1}{100}\sum_{k=1}^{100} p(y_t=j|x_t,\hat w)\right) \log \left(\frac{1}{100}\sum_{k=1}^{100} p(y_t=j|x_t,\hat w)\right) \\
    &=: \tilde{\mathbb{H}}_t.
    \end{aligned}
\end{equation*}
$$

Essentially, $ j$ iterates over each class and summarizes the average level of uncertainty for outcomes of that class. The function is minimized when the model is certain––when one class has probability 1 and all others are 0. The function is maximized when the model is very uncertain––probability is uniform across the classes. Also observe that our earlier notation $\hat p_{j,t}$ is shorthand for $\frac{1}{100}\sum_{k=1}^{100} p(y_t=j|x_t,\hat w)$.

Using this metric, we upsize our positions if our model is certain and downsize our positions if the model is uncertain. More specifically, we still go long or short if 
$ \hat{p}_{1,t} > \alpha$ or 
$ \hat{p}_{-1,t} > \alpha$, respectively, but we upsize our positions to 
$ 1.5 \times \mu $ if 
$ \tilde{\mathbb{H}}_t<\beta_1 $, keep our size 
$\mu$ if 
$ \beta_1< \tilde{\mathbb{H}}_t < \beta_2 $, downsize to 
$0.5 \times \mu$ if 
$ \tilde{\mathbb{H}}_t>\beta_2 $, and exit the current position if 
$ \tilde{\mathbb{H}}_t < \beta_2 $[<sub>[2]</sub>](#ref2). We fix values for 
$\alpha$ and 
$\beta_2$ and test different values for 
$\beta_1$.

### Results
Now comes the question of how we should compare these strategies in terms of profit and risk. 

Since each strategy returns different transaction volumes, we standardize profits to properly compare profitability. And while the Sharpe ratio is a popular measure of risk in a portfolio or strategy, it deems large positive and negative returns to be equally risky, so we follow BDLOB[<sub>[2]</sub>](#ref2) in using the Downward Deviation ratio 
$ \text{DDR} = \frac{\mathbb{E}(R_t)}{\text{DD}_T} $ as our risk measure, where 
$ \mathbb{E}(R_t) $ is the average return per timestamp and 
$ \text{DD}_T = \sqrt{\frac{1}{T} \sum_{t=1}^T \text{min}(R_t,0)^2}$ measures the deviation of negative returns. DDR has the desired property of penalizing negative returns and rewarding positive returns.

Unfortunately, the models were both very poor in their ability to generate profits, but this should come as no surprise. Recall that our models predict downward moves almost identically poorly, so it is reasonable to believe that the model is doomed to behave poorly in a downward trending regime. And since the Bitcoin mid price dropped 1% over the duration of the trading period, our understanding of the model justify the results we see.

Although the losses are very disappointing, we still observe steadiness in the Bayesian strategy relative to the softmax strategy, telling us the concept of incorporating model uncertainty is important. Also, normalized profits clearly illustrate the benefit that training on order flow offered over order flow imbalance.

Proper downsampling would have been a nice add in our training procedure in order to help our model better predict the downward moves that were observed in the out-of-sample data.

# Model Diagnostics
In acccordance with the Box-Jenkins approach, we test the fitted models' residuals $\{\hat{u}_i\}_{i=1}^{T}$ for any autocorrelation. If true, there is statistical evidence that the model is underfitting, and we should increase the lag parameter of our sequential model and re-train. If false, we accept the model residuals to be white noise. 

We compute our residuals as the cross-entropy of the classification problem at each timestamp, which we define by
$$ \hat{u}_i=-\sum_{j=-1}^{1}y_i(j)\log\hat{y}_i(j) $$
for 
$i \in \{1,...,T\}$, where 
$y_i$ is the one-hot encoded 3-variable vector of the true 1-step movement, 
$\hat{y}_i$ is our model's unrounded prediction of that encoding, and $T$ is the number of observations.

Letting 
$\hat{\tau}_i$ be the sample autocorrelations of the residuals and $m$ to be a maximum lag to test, we use the Ljung-Box statistic
$$ Q(m) = T(T+2)\sum_{l=1}^{m}\frac{\hat{\tau}_l^2}{T-l} $$
as our test statistic for the null hypothesis 
$H_0: \tau_1=...=\tau_m=0$ versus the alternative 
$H_a: \tau_i \neq 0$ for some 
$i \in \{1,...,m\} $. For large $T$, the statistic is chi-squared distributed with $m$ degrees of freedom, and we reject the null in favor of the alternative if the test statistic is greater than the critical value of the corresponding chi-squared distribution at the 99% confidence level. 

We unfortunately found evidence supporting the conclusion that both models underfit the training data, so increasing the lag parameter and redoing the training and diagnostics until we no longer underfit would be a necessary next step, but we omit it for brevity. Also, Kolm et al[<sub>[4]</sub>](#ref4) found that different choices for the lag parameter had little impact on the performance of the CNN-LSTM model for their regression problem, so perhaps this is as good as we can get. And as a third point, the fact that the model is underfitting should come as no surprise, since this really tells us that our model is too simple for the data. Financial data, even such stationary processes as order flow and order flow imbalance, are incredibly complex, so it's hard to expect any interpretable model to well-fit the input data. In this truth lies one of the problems of deep learning in finance.

## References
1. <span id="ref1">Z. Zhang, S. Zohren, and S. Roberts. DeepLOB: Deep Convolutional Neural Networks for Limit Order Books *IEEE Transactions On Signal Processing* 67(11):3001–3012, 2019.<br>
2. <span id="ref2">Z. Zhang, S. Zohren, and S. Roberts. BDLOB: Bayesian Deep Convolutional Neural Networks For Limit Order Books. *arXiv preprint arXiv:1811.10041*, 2018.<br>
3. <span id="ref3">R. Cont, A. Kukanov, and S. Stoikov. The Price Impact Of
Order Book Events. *Journal Of Financial Econometrics* 12(1):47–88, 2014.<br>
4. <span id="ref4">P. Kolm, J. Turiel, and N. Westray. Deep Order Flow Imbalance: Extracting Alpha At Multiple Horizons From The Limit Order Book. *Available at SSRN 3900141*, 2021.<br>
5. <span id="ref5">H. Lütkepohl. *New Introduction to Multiple Time Series Analysis*. Springer, 2005.<br> 
6. <span id="ref6">M. F. Dixon, I. Halperin, and P. Bilokon. *Machine Learning in Finance: From Theory to Practice*. Springer, 2020.<br>
7. <span id="ref7">S. Hochreiter and J. Schmidhuber. Long Short-Term Memory. *Neural Computation* 9(8):1735–1780, 1997.<br>
8. <span id="ref8">A. Ntakaris, M. Magris, J. Kanniainen, M. Gabbouj, and A. Iosifidis. Benchmark Dataset For Mid-Price Forecasting Of Limit Order Book Data With Machine Learning Methods. *Journal of Forecasting* 37(8):852–866, 2018.<br>
9. <span id="ref9">K. Yang and C. Shahabi. On the Stationarity of Multivariate Time Series for Correlation-Based Data Analysis. *Fifth IEEE International Conference on Data Mining* 1-4, 2005.<br>
10. <span id="ref10">Y. Gal. *Uncertainty in Deep Learning*. Phd Thesis, University of Cambridge, 2016.<br>
11. <span id="ref11">O. Surakhi, M. A. Zaidan, P. L. Fung, N. H. Motlagh, S. Serhan, M. AlKhanafseh, R. M. Ghoniem, and T. Hussein. Time-Lag Selection for Time-Series Forecasting Using Neural Network and Heuristic Algorithm. *Electronics* 10(20), 2021.<br>

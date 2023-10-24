# Deep Learning Interview Questions


Topics
---

- [Bayesian Probability]()


## Bayesian Probability

Contents
----

- Expectation and Variance
- Conditional Probability
- Bayes Rule 
- Maximum Likelihood Estimation
- Fisher Information
- Posterior&prior predictive distributions 
- Conjugate priors
    * The Beta-Binomial distribution
- Bayesian Deep Learning 


---

1. Define what is meant by a Bernoulli trial.

---

2. The binomial distribution is often used to model the probability that $k$ out of a group of $n$
objects bare a specific characteristic. Define what is meant by a binomial random variable $X$.

---

3. What does the following shorthand stand for?

$$
X ∼ Binomial(n, p)
$$

---

4. Find the probability mass function (PMF) of the following random variable:

$$
X ∼ Binomial(n, p)
$$

---

5. Answer the following questions:
    1. Define what is meant by (mathematical) expectation.
    2. Define what is meant by variance.
    3. Derive the expectation and variance of a the binomial random variable $X ∼ Binomial(n, p)$ in terms of $p$ and $n$.

---

6. Proton therapy (PT) is a widely adopted form of treatment for many types of cancer.
A PT device which was not properly calibrated is used to treat a patient with pancreatic cancer (Fig. 3.1). As a result, a PT beam randomly shoots $200$ particles independently and correctly hits cancerous cells with a probability of $0.1$.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-1.png" alt= "Histopathology for pancreatic cancer cells" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">Histopathology for pancreatic cancer cells</td>
</tr>
</table>

1. Find the statistical distribution of the number of correct hits on cancerous cells in the described experiment. What are the expectation and variance of the corresponding random variable?
2. A radiologist using the device claims he was able to hit exactly 60 cancerous cells. How likely is it that he is wrong?

---

7. Given two events $A$ and $B$ in probability space $H$, which occur with probabilities $P(A)$ and $P(B)$, respectively:
    1. Define the conditional probability of $A$ given $B$. Mind singular cases. 
    2. Annotate each part of the conditional probability formulae.
    3. Draw an instance of Venn diagram, depicting the intersection of the events $A$ and $B$. Assume that $A  \cup B = H$.

---

8. Bayesian inference amalgamates data information in the likelihood function with known prior information. This is done by conditioning the prior on the likelihood using the Bayes formulae. Assume two events A and B in probability space $H$, which occur with probabilities $P(A)$ and $P(B)$, respectively. Given that $A \cup B = H$, state the Bayes formulae for this case, interpret its components and annotate them.

---

9. Define the terms likelihood and log-likelihood of a discrete random variable X given a fixed parameter of interest $\gamma$. Give a practical example of such scenario and derive its likelihood and log-likelihood.

---

10. Define the term prior distribution of a likelihood parameter $\gamma$ in the continuous case.

---

11. Show the relationship between the prior, posterior and likelihood probabilities.

---

12. In a Bayesian context, if a first experiment is conducted, and then another experiment is followed, what does the posterior become for the next experiment?

---

13. What is the condition under which two events $A$ and $B$ are said to be statistically independent?

---

14. In an experiment conducted in the field of particle physics (Fig. 3.2), a certain particle may be in two distinct equally probable quantum states: integer spin or half-integer spin. It is well-known that particles with integer spin are bosons, while particles with half-integer spin are fermions.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-2.png" alt= "Bosons and fermions: particles with half-integer spin are fermions" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">Bosons and fermions: particles with half-integer spin are fermions</td>
</tr>
</table>

A physicist is observing two such particles, while at least one of which is in a half-integer state. What is the probability that both particles are fermions?

---

15. During pregnancy, the Placenta Chorion Test is commonly used for the diagnosis of hereditary diseases (Fig. 3.3). The test has a probability of $0.95$ of being correct whether or not a hereditary disease is present.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-3.png" alt= "Foetal surface of the placenta" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">Foetal surface of the placenta</td>
</tr>
</table>
It is known that $1\%$ of pregnancies result in hereditary diseases. Calculate the probability of a test indicating that a hereditary disease is present.

---

16. The Dercum disease is an extremely rare disorder of multiple painful tissue growths.
In a population in which the ratio of females to males is equal, 5% of females and 0.25% of males have the Dercum disease (Fig. 3.4).
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-4.png" alt= "The Dercum disease<" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">The Dercum disease<</td>
</tr>
</table>

A person is chosen at random and that person has the Dercum disease. Calculate the probability that the person is female.

---

17. There are numerous fraudulent binary options websites scattered around the Internet, and for every site that shuts down, new ones are sprouted like mushrooms. A fraudulent AI based stock-market prediction algorithm utilized at the New York Stock Exchange, (Fig. 3.6) can correctly predict if a certain binary option shifts states from 0 to 1 or the other way around, with $85\%$ certainty.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-5.png" alt= "The New York Stock Exchange" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">The New York Stock Exchange</td>
</tr>
</table>
A financial engineer has created a portfolio consisting twice as many $state-1$ options then $state-0$ options. A stock option is selected at random and is determined by said algorithm to be in the state of $1$. What is the probability that the prediction made by the AI is correct?

---

18. In an experiment conducted by a hedge fund to determine if monkeys (Fig. 3.6) can
outperform humans in selecting better stock market portfolios, 0.05 of humans and 1 out of 15 monkeys could correctly predict stock market trends correctly.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-6.png" alt= "Hedge funds and monkeys" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">Hedge funds and monkeys</td>
</tr>
</table>

From an equally probable pool of humans and monkeys an “expert” is chosen at random. When tested, that expert was correct in predicting the stock market shift. What is the probability that the expert is a human?

---

19. During the cold war, the U.S.A developed a speech to text (STT) algorithm that could theoretically detect the hidden dialects of Russian sleeper agents. These agents (Fig. 3.7), were trained to speak English in Russia and subsequently sent to the US to gather intelligence. The FBI was able to apprehend ten such hidden Russian spies and accused them of being "sleeper" agents.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-7.png" alt= "Dialect detection" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">Dialect detection</td>
</tr>
</table>

The Algorithm relied on the acoustic properties of Russian pronunciation of the word *(v-o-k-s-a-l)* which was borrowed from English *V-a-u-x-h-a-l-l*. It was alleged that it is impossible for Russians to completely hide their accent and hence when a Russian would say *V-a-u-x-h-a-l-l*, the algorithm would yield the text *v-o-k-s-a-l*. To test the algorithm at a diplomatic gathering where $20\%$ of participants are Sleeper agents and the rest Americans, a data scientist randomly chooses a person and asks him to say *V-a-u-x-h-a-l-l*. A single letter is then chosen randomly from the word that was generated by the algorithm, which is observed to be an "l". What is the probability that the person is indeed a Russian sleeper agent?

---

20. During World War II, forces on both sides of the war relied on encrypted communications. The main encryption scheme used by the German military was an Enigma machine, which was employed extensively by Nazi Germany. Statistically, the Enigma machine sent the symbols X and Z Fig. (3.8) according to the following probabilities:

$$

P(X) = \frac{2}{9} \\ 
\\ 
P(Z) = \frac{7}{9}

$$
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-8.png" alt= "The Morse telegraph code" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">The Morse telegraph code</td>
</tr>
</table>

In one incident, the German military sent encoded messages while the British army used countermeasures to deliberately tamper with the transmission. Assume that as a result of the British countermeasures, an X is erroneously received as a Z (and mutatis mutandis) with a probability $\frac{1}{7}$. If a recipient in the German military received a Z, what is the probability that a Z was actually transmitted by the sender?

---

21.  What is likelihood function of the independent identically distributed (i.i.d) random variables:
$X_1,··· ,X_n$ where $X_i ∼ binomial(n, p)$, $∀i ∈ [1,n]$, and where p is the parameter of interest?

---

22. How can we derive the maximum likelihood estimator (MLE) of the i.i.d samples $X_1, · · · , X_n$ introduced in above question?

---

23.  What is the relationship between the likelihood function and the log-likelihood function?

---

24.  Describe how to analytically find the MLE of a likelihood function?

---

25. What is the term used to describe the first derivative of the log-likelihood function?

---

26.  Define the term Fisher information.

---

27. The 2014 west African Ebola (Fig. 9.10) epidemic has become the largest and fastest spreading outbreak of the disease in modern history with a death tool far exceeding all past outbreaks combined. Ebola (named after the Ebola River in Zaire) first emerged in 1976 in Sudan and Zaire and infected over 284 people with a mortality rate of 53%.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-9.png" alt= "The Ebola virus" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">The Ebola virus</td>
</tr>
</table>

This rare outbreak, underlined the challenge medical teams are facing in containing epidemics. A junior data scientist at the center for disease control (CDC) models the possible spread and containment of the Ebola virus using a numerical simulation. He knows that out of a population of k humans (the number of trials), x are carriers of the virus (success in statistical jargon). He believes the sample likelihood of the virus in the population, follows a Binomial distribution:

$$
L(\gamma | y) = (n, y)\gamma^y(1-\gamma)^{n-y}, \ \gamma \epsilon [0, 1], \  y = 1,2,...n
$$

As the senior researcher in the team, you guide him that his parameter of interest is $γ$, the proportion of infected humans in the entire population. The expectation and variance of the binomial distribution are:
$$
E(y|γ, n) = nγ, V (y|γ, n) = nγ(1 − γ) 
$$
Answer the following; for the likelihood function of the form $L_x(γ)$:
1. Find the log-likelihood function $l_x(γ) = ln L_x(γ)$.
2. Find the gradient of $l_x(γ)$.
3. Find the Hessian matrix $H(γ)$.
4. Find the Fisher information $I(γ)$.
5. In a population spanning $10,000$ individuals, $300$ were infected by Ebola. Find the MLE for γ and the standard error associated with it.

---

28. In this question, you are going to derive the Fisher information function for several distributions. Given a probability density function (PDF) $f(X|γ)$, you are provided with the following definitions:
    1. The natural logarithm of the PDF $lnf(X|γ) = Φ(X|γ)$.
    2. The first partial derivative $Φ′(X|γ)$.
    3. The second partial derivative $Φ′′(X|γ)$.
    4. The Fisher Information for a continuous random variable:
$$
I(γ) = −Eγ[Φ′(X|γ)]
$$
Find the Fisher Information $I(γ)$ for the following distributions:
    1. The Bernoulli Distribution $X ∼ B(1, γ)$.
    2. The Poisson Distribution $X ∼ Poiss(θ)$.

---

29. 1. **True or False**: The Fisher Information is used to compute the Cramer-Rao bound on the variance of any unbiased maximum likelihood estimator.
    2. **True or False**: The Fisher Information matrix is also the Hessian of the symmetrized KL divergence.

---

30. 1. Define the term posterior distribution.
    2. Define the term prior predictive distribution.

---

31. Let y be the number of successes in 5 independent trials, where the probability of success is θ in each trial. Suppose your prior distribution for θ is as follows: $P(θ = 1/2) = 0.25, P (θ = 1/6) = 0.5, and P (θ = 1/4) = 0.25$.
    1. Derive the posterior distribution $p(θ|y)$ after observing y. 
    2. Derive the prior predictive distribution for y.

---

32. 1. Define the term conjugate prior.
    2. Define the term non-informative prior.

---

33. Prove that the family of beta distributions is conjugate to a binomial likelihood, so that if a prior is in that family then so is the posterior. That is, show that:

$$
x ∼ Ber(γ), γ ∼ B(α,β) ⇒ γ|x ∼ B(α′,β′)
$$

For instance, for h heads and t tails, the posterior is:

$$
B(h + α,t + β)
$$

---

34. A recently published paper presents a new layer for a new Bayesian neural network (BNN). The layer behaves as follows. During the feed-forward operation, each of the hidden neurons $H_n , n ∈ 1, 2$ in the neural network (Fig. 3.10) may, or may not fire independently of each other according to a known prior distribution.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-10.png" alt= "Likelihood in a BNN model" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">Likelihood in a BNN model</td>
</tr>
</table>

The chance of firing, γ, is the same for each hidden neuron. Using the formal definition, calculate the likelihood function of each of the following cases:
1. The hidden neuron is distributed according to $X ∼ binomial(n, γ)$ random variable and fires with a probability of $γ$. There are 100 neurons and only 20 are fired.
2. The hidden neuron is distributed according to $X ∼ Uniform(0,γ)$ random variable and fires with a probability of $γ$.

---

35. Your colleague, a veteran of the Deep Learning industry, comes up with an idea for for a BNN layer entitled OnOffLayer. He suggests that each neuron will stay on (the other state is off) following the distribution $f(x) = e^{−x} \ for \ x > 0 \ and \ f(x) = 0 \ otherwise (Fig. 3.11)$. $X$ indicates the time in seconds the neuron stays on. In a BNN, 200 such neurons are activated independently in said OnOffLayer. The OnOffLayer is set to off (e.g. not active) only if at least 150 of the neurons are shut down. Find the probability that the OnOffLayer will be active for at least 20 seconds without being shut down.

<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-11.png" alt= "OnOffLayer in a BNN model" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">OnOffLayer in a BNN model</td>
</tr>
</table>

---

36. A Dropout layer(Fig. 3.12) is commonly used to regularize a neural network model by randomly equating several outputs (the crossed-out hidden node H) to 0.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-12.png" alt= "A Dropout layer (simplified form)" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">A Dropout layer (simplified form)</td>
</tr>
</table>
For instance, in PyTorch, a Dropout layer is declared as follows:

```python

import torch
import torch.nn as nn
nn.Dropout(0.2)

```

Where nn.Dropout(0.2) (Line #3 in 3.1) indicates that the probability of zeroing an element is 0.2.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-13.png" alt= "A Bayesian Neural Network Model" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">A Bayesian Neural Network Model</td>
</tr>
</table>

A new data scientist in your team suggests the following procedure for a Dropout layer which is based on Bayesian principles. Each of the neurons $θ_n$ in the neural network in (Fig. 8.33) may drop (or not) independently of each other exactly like a Bernoulli trial.

During the training of a neural network, the Dropout layer randomly drops out outputs of the previous layer, as indicated in (Fig. 3.12). Here, for illustration purposes, all two neurons are dropped as depicted by the crossed-out hidden nodes $H_n$.

You are interested in the proportion θ of dropped-out neurons. Assume that the chance of drop-out, $θ$, is the same for each neuron (e.g. a uniform prior for $θ$). Compute the posterior of $θ$.

---

37. A new data scientist in your team, who was formerly a Quantum Physicist, suggests the following procedure for a Dropout layer entitled QuantumDrop which is based on Quantum principles and the Maxwell Boltzmann distribution. In the Maxwell-Boltzmann distribution, the likelihood of finding a particle with a particular velocity v is provided by:

$$
n(v)dv = \frac{4\pi N}{V}(\frac{m}{2\pi kT})^{\frac{3}{2}}v^2e^{-\frac{mv^2}{2kT}}dv
$$

<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-14.png" alt= "The Maxwell-Boltzmann distribution" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">The Maxwell-Boltzmann distribution</td>
</tr>
</table>

In the suggested QuantumDrop layer (3.15), each of the neurons behaves like a molecule and is distributed according to the Maxwell-Boltzmann distribution and fires only when the most probable speed is reached. This speed is the velocity associated with the highest point in the Maxwell distribution (3.14). Using calculus, brain power and some mathem- atical manipulation, find the most likely value (speed) at which the neuron will fire.
<table align='center'>
<tr>
<td align="center">
    <img src="img/prob-15.png" alt= "A QuantumDrop layer" style="max-width:70%;" />
</td>
</tr>
<tr>
<td align="center">A QuantumDrop layer</td>
</tr>
</table>

---
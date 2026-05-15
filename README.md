# Importance-Sampling-with-Adapted-Exponential-Families
This Repository has the Code my Bachelors Thesis "Importance Sampling with Adapted Exponential Families".  All Information can be read in the thesis itself and some summery is down below.

## Important Code 
`class_IS_optimizer.py` and `class_natural_parameter.py` are the heart of this project. The best example on how to use them is `example_BlackScholes_Asian_Spread_Call.ipynb`.
Down below is now a summery of the theory and some notes on how to use the code.

# Importance Sampling with Adapted Exponential Families

This repository implements an Importance Sampling (IS) framework based on **adapted exponential families**. The goal is to approximate expectations of path-dependent functionals

$$
\mu_f := \mathbb{E}_{\bar p}[f(X_1,\dots,X_S)]
$$

where

$$
X := (X_1,\dots,X_S), \qquad X_s \in \mathbb{R}^d \quad \forall s =1,...,S ,
$$

and $\bar p$ denotes the original sampling density of the full path $X$. A standard Monte Carlo estimator is given by

$$
\mu_f^{(M)} := \frac{1}{M}\sum_{m=1}^M f(X^{(m)}),
\qquad X^{(m)} \sim \bar p.
$$

By the strong law of large numbers for $M\to\infty$  ,
$$
\mu_f^{(M)} \rightarrow \mu_f \text{ a.s.}.
$$

However, the variance of this estimator can be large, especially for rare-event or path-dependent problems. Importance Sampling tries to reduce this variance by sampling from another density $q$ instead of $\bar p$.

## Importance Sampling

Let $q$ be an importance density such that it is viable wrt to $\bar p$

Then

$$
\mu_f
= \int f(x)\bar p(x)dx
= \int f(x)\frac{\bar p(x)}{q(x)}q(x)\,dx
= \mathbb{E}_q\left[I(X)f(X)\right],
$$

where

$$
I(x) := \frac{\bar p(x)}{q(x)}
$$

is the likelihood ratio. The corresponding IS estimator is

$$
\mu_{f,I}^{(M)}
:= \frac{1}{M}\sum_{m=1}^M I(X^{(m)})f(X^{(m)}),
\qquad X^{(m)} \sim q.
$$

Both $\mu_f^{(M)}$ and $\mu_{f,I}^{(M)}$ estimate the same expectation $\mu_f$, but their variances are generally different. The objective is therefore to choose $q$ such that

$$
Var_q(\mu_{f,I}^{(M)})
\leq
Var_{\bar p}(\mu_f^{(M)}).
$$
ie we want to minimize the variance.

## Exponential Families

A basic exponential family has densities of the form

$$
q(x \mid z)=q_0(x)\exp\left(z^\top T(x) - A(z)\right),
$$

where

- $q_0$ is the base density,
- $T(x)$ is the sufficient statistic,
- $z$ is the natural parameter,
- $A(z)$ is the log-partition function,

with

$$
A(z)=\log\left(\int q_0(x)\exp(z^\top T(x))dx\right).
$$

The function $A(z)$ normalizes the density, i.e.

$$
\int g(x\mid z)dx = 1.
$$

## Adapted Exponential Families

The main idea of this project is to construct the importance density sequentially. Let

$$
X = (X_1,\dots,X_S)
$$

and let $(\mathcal{F}_s)_{s=0}^S$ be the natural filtration

$$
\mathcal{F}_s := \sigma(X_1,\dots,X_s).
$$

An adapted exponential family is constructed as

$$
q(x\mid z)=\prod_{s=1}^S q^{(s)}(x_s \mid z^{(s)}),
$$

where each component has exponential-family form

$$
q^{(s)}(x_s \mid z^{(s)})=q_0^{(s)}(x_s)\exp\left((z^{(s)})^\top T^{(s)}(x_s)-A^{(s)}(z^{(s)})\right).
$$

The key adaptation condition is that

$$
z^{(s)} \text{ is } \mathcal{F}_{s-1}\text{-measurable}.
$$

Equivalently,

$$
z^{(s)}=z^{(s)}(X_1,\dots,X_{s-1}).
$$

Thus, the natural parameter used to sample $X_s$ may depend on the previously sampled values $X_1,\dots,X_{s-1}$, but not on the current or future values. In particular, $z^{(1)}$ is deterministic because $\mathcal{F}_0$ is trivial.

This gives a sequential importance sampler:

1. Choose $z^{(1)}$.
2. Sample $X_1 \sim q^{(1)}(\cdot \mid z^{(1)})$.
3. Compute $z^{(2)}(X_1)$.
4. Sample $X_2 \sim q^{(2)}(\cdot \mid z^{(2)}(X_1))$.
5. Continue until $X_S$ is sampled.
6. Compute the likelihood ratio

$$
I(X)=\frac{\bar p(X)}{q(X\mid z)}.
$$

## Defining the Natural Parameter

The thesis explains how the natural parameter is defined in more detail. See chapter 4.3 "Creating an Adapted Exponential Family for IS" (page 29 to be more accurate). 

## Code Implementation Natural Parameter: 

`class_natural_parameter_components.py` allows us to define the natural parameter $z$ accordingly. In `example_BlackScholes_Asian_Spread_Call.ipynb` we can see under `list_of_z = natural_parameter_definer(typ="stdNormal", beta=5).create_z_normal(d, S, combination_typ="mean_in_typ")` that the natural parameter is fully defined as a list of functions which take the samples, ie
$$
z^{(s)}=z^{(s)}(X_1,\dots,X_{s-1}).
$$

## Variance Optimization

The variance-reduction problem can be formulated through the second moment

$$
\mathbb{E}_q\left[\left(
\frac{\bar p(X)}{q(X\mid z)}f(X)
\right)^2
\right].
$$

Ignoring the constant $\mu_f^2$, minimizing this quantity is equivalent to minimizing the IS variance.

Since direct integration is usually not feasible, the thesis replaces the integral by a Monte Carlo approximation using training samples

$$
X^{(m)} \sim \bar p.
$$

This leads to the empirical objective

$$
\frac{1}{M}
\sum_{m=1}^M
\bar p(X^{(m)})
\left(q(X^{(m)}\mid z)\right)^{-1}
\left(f(X^{(m)})\right)^2.
$$

To make the optimization more practical, the natural parameter is written with additional weights $\alpha$. For every time step $s$,

$$
z_\alpha^{(s)}
:=
\alpha^{(s)} \odot z^{(s)},
$$

where $\odot$ denotes component-wise multiplication. Then

$$
z_\alpha := (z_\alpha^{(1)},\dots,z_\alpha^{(S)}).
$$

The optimization problem becomes

$$
\min_{\alpha \in \mathcal{A}}
\sum_{m=1}^M
\bar p(X^{(m)})
\left(q(X^{(m)}\mid z_\alpha)\right)^{-1}
\left(f(X^{(m)})\right)^2,
$$

where $\mathcal{A}$ is chosen to be convex.

The density used in the objective is

$$
q(X^{(m)}\mid z_\alpha)=q_0(X^{(m)})\exp\left(\sum_{s=1}^S(z_\alpha^{(s)})^\top T^{(s)}(X_s^{(m)})-A^{(s)}(z_\alpha^{(s)})\right).
$$

The convexity of this objective follows from the convexity of the log-partition function $A$ and the exponential structure of the family.

## Code Implementation Natural Parameter: 

`class_IS_optimizer.py` to find the best $\alpha$ and therefore the best adapted exponential family $q$. In `example_BlackScholes_Asian_Spread_Call.ipynb` we can see under `opti = IS_optimizer(list_of_z, samples_test, p_joint, f)` what inputs are required and how to to use it.


## Algorithmic Structure

The implemented workflow is:

1. Define the target density $\bar p$ and payoff/function $f$.
2. Choose an exponential family type, for example:
   - multivariate normal distribution,
   - exponential distribution.
3. Define adapted natural parameters

$$
z^{(s)}(X_1,\dots,X_{s-1}).
$$

4. Generate training samples from the original density $\bar p$.
5. Solve the convex optimization problem for $\alpha$.
6. Construct the adapted importance density $q(\cdot \mid z_\alpha)$.
7. Sample paths from $q$.
8. Estimate

$$
\mu_f
\approx
\frac{1}{M}
\sum_{m=1}^M
\frac{\bar p(X^{(m)})}{q(X^{(m)}\mid z_\alpha)}
f(X^{(m)}).
$$

## Tested Models

The framework is tested on two main examples.

### Model 1: Black-Scholes Model with Asian Spread Call Option

The first model applies the method to a Black-Scholes setting with a path-dependent Asian spread call payoff. The payoff has the form

$$
f(X)=e^{-rT}\max\left(0,\bar S_1 - \bar S_2 - K\right),
$$

where $\bar S_1$ and $\bar S_2$ are arithmetic averages of two simulated asset paths.

For this model, the adapted exponential family is based on a multivariate normal distribution. The results show that the IS approximation can improve the Monte Carlo estimate and reduce variance for many choices of adapted natural parameters.

### Model 2: Poisson Process with Exponential Distribution

The second model uses exponential distributions in the context of a Poisson-process-related example. The results are more mixed. In particular, the performance can deteriorate for larger values of $S$, suggesting that the quality of the adapted density is strongly problem-dependent.

## Main Takeaway and Intuitive Idea

This project explores Importance Sampling with adapted exponential families as a flexible variance-reduction method for path-dependent Monte Carlo problems. The central idea is to replace a fixed importance density by a sequentially adapted one, where the natural parameter at time $s$ may depend on the previously sampled values.

In my head this method allows also to weigh the samples in there importance. For example, consider the $X_1,...,X_5 \sim N(0,1)$ iid, then $x_1, x_2, x_3, x_4 \in [100,200]$ would be highly unlikely and could corrupt the end result, ie the sample vector $(x_1,...x_5)$ is most likely just bad, even before we know what $x_5$ is. So what if we punish such behavior by constructing $z$ so that the sample vector $(x_1,...x_5)$ has (almost) no weight when producing such bad results, fe choose $z^{(5)}$ such that $z^{(5)}(x_1,...,x_4)$ has no impact on the approximation.
For course all that needs more experimentation and research.

In this lecture, the prof talked about basic statistics and programming of data analytics, but I don't think she was specific enough, so I try to expand on several topics.



# 1. Poisson Distribution

## 1.1 Introduction to problem

Consider the problem: Suppose a call center representative receives an average of 

Here is a popular tutorial you can find online: [Poisson Distribution — Intuition, Examples, and Derivation](https://towardsdatascience.com/poisson-distribution-intuition-and-derivation-1059aeab90d)

But I think the tutorial is just...wrong. The problem considered is:

```
Every week, on average, 17 people clap for my blog post. 
I’d like to predict the number of people who will clap next week, say, because I get paid weekly based on those numbers.
What is the probability that exactly 20 people (or 10, 30, 50, etc.) will clap for the blog post next week?
```

And the author tries to use Binomial Distribution to solve the problem, then derive Poisson Distribution from Binomial. This problem is a good example of the usage of Poisson Distribution, but the issue is: you cannot use Binomial distribution to solve the problem, because you cannot find an independent trial of the same probability. The author just takes the limit of a mysterious $n$ which doesn't have any physical meaning and derive the Poisson Distribution.

Now, how could we derive Poisson Distribution from scratch? Students get very confused about Poisson Distribution because a typical beginner-lever Statistical book will skip the derivation of Poisson Distribution, and only give the result, so they don't know where the Poisson Distribution is from.

## 1.2 Derivation of Poisson Distribution

Here is the true derivation of Poisson Distribution: [Derivation of the Poisson distribution](https://www.pp.rhul.ac.uk/~cowan/stat/notes/PoissonNote.pdf)

Now I just try to interpret in a way that an engineering student can understand.

Firstly, we need to specify what we are trying to derive:

**Theorem 1. Given a time interval, a parameter $\lambda>0$, for a discrete random variable $X$ whose expected value is $\lambda$, and it is guaranteed that events are independent of each other, then the probability mass function(PMF) is given by $Pr(X=k)=\frac{\lambda^ke^{-\lambda}}{k!}$**

This is different from a typical formulation of Poisson Distribution that you may see in a textbook:

**Theorem 2. A discrete random variable $X$ is said to have a Poisson distribution, with parameter $\lambda > 0$, if it has a probability mass function given by:**
$$
f(k;\lambda) = Pr(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$
**where**

**$k$ is the number of occurrences ($k = 0, 1, 2, \ldots$),**
**$e$ is Euler's number ($e \approx 2.71828$),**
**$k!$ is the factorial function.**

But I believe Theorem 1 is exactly what we need to derive, in order to figure out where the Poisson Distribution is from, and to use it to model the real problems. As learners, I believe your confusion mainly lies in where the mysterious $\frac{\lambda^k e^{-\lambda}}{k!}$comes from. 

$\lambda$
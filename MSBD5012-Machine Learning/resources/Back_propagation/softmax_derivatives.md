Certainly. Deriving the gradient for the softmax layer can indeed be a bit tricky. We'll focus on a single example first, and then extend it to a batch of examples.

### Softmax Definition

Given an input vector $$ \mathbf{z} $$, the softmax function $$ S(z)_i $$ for the $$i$$-th component is defined as:

$$
S(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

### Cross-Entropy Loss Definition

The cross-entropy loss $$ J $$ for true label $$ y $$ and predicted probabilities $$ \mathbf{a} $$ is:

$$
J = -\log(a_y) = -\log(S(z)_y)
$$

### Objective: Derive $$ \frac{\partial J}{\partial z_i} $$

We want to find how much the loss $$ J $$ changes with a small change in $$ z_i $$, where $$ i $$ could be the true label $$ y $$ or any other class. For convenience, let $$ \Sigma = \sum_{j=1}^{K} e^{z_j} $$.

#### Case 1: $$ i = y $$

$$
\begin{align*}
\frac{\partial J}{\partial z_y} &= \frac{\partial}{\partial z_y} (-\log(S(z)_y)) \\
&= -\frac{1}{S(z)_y} \frac{\partial S(z)_y}{\partial z_y} \\
&= -\frac{1}{S(z)_y} \frac{\partial}{\partial z_y} \left( \frac{e^{z_y}}{\Sigma} \right) \\
&= -\frac{1}{S(z)_y} \left( \frac{e^{z_y} \Sigma - e^{z_y} e^{z_y}}{\Sigma^2} \right) \\
&= -\frac{1}{S(z)_y} \frac{e^{z_y} (\Sigma - e^{z_y})}{\Sigma^2} \\
&= -\frac{1}{S(z)_y} \frac{e^{z_y}}{\Sigma} \frac{\Sigma - e^{z_y}}{\Sigma} \\
&= -S(z)_y (1 - S(z)_y) \\
&= S(z)_y - 1
\end{align*}
$$

#### Case 2: $$ i \neq y $$

$$
\begin{align*}
\frac{\partial J}{\partial z_i} &= -\frac{1}{S(z)_y} \frac{\partial S(z)_y}{\partial z_i} \\
&= -\frac{1}{S(z)_y} \frac{\partial}{\partial z_i} \left( \frac{e^{z_y}}{\Sigma} \right) \\
&= -\frac{1}{S(z)_y} (-\frac{e^{z_y} e^{z_i}}{\Sigma^2}) \\
&= \frac{1}{S(z)_y} \frac{e^{z_y} e^{z_i}}{\Sigma^2} \\
&= S(z)_i S(z)_y \\
&= S(z)_i (S(z)_y - 1)
\end{align*}
$$

### Final Result

Putting it all together, we have:

$$
\frac{\partial J}{\partial z_i} = 
\begin{cases} 
S(z)_i - 1 & \text{if } i = y \\
S(z)_i & \text{if } i \neq y
\end{cases}
$$

This can be succinctly written in vector form as $$ \Delta = \mathbf{S(z)} - \mathbf{y} $$, where $$ \mathbf{y} $$ is the one-hot encoded label vector, and $$ \Delta $$ is the derivative of the loss with respect to the softmax inputs $$ \mathbf{z} $$.

For a batch of $$ N $$ examples, this formula naturally extends to matrix operations.
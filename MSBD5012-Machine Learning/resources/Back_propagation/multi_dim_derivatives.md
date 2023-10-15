Certainly! To adapt the explanation and notation to account for a batch of \( N \) samples, the forward and backward propagation steps can be modified. Here, \( X \) will be a matrix of dimensions \( [N, D] \), and \( W \) will be of shape \( [\text{dim\_inputs}, \text{dim\_outputs}] \).

### Notation:

- \( L \): Output layer
- \( Z^{[l]} \): Pre-activation matrix at layer \( l \) (shape \( [N, \text{dim\_outputs}^{[l]}] \))
- \( A^{[l]} \): Activation matrix at layer \( l \) (shape \( [N, \text{dim\_outputs}^{[l]}] \))
- \( W^{[l]} \): Weights for layer \( l \) (shape \( [\text{dim\_inputs}^{[l]}, \text{dim\_outputs}^{[l]}] \))
- \( b^{[l]} \): Biases for layer \( l \) (shape \( [1, \text{dim\_outputs}^{[l]}] \))
- \( \Delta^{[l]} \): Error term for layer \( l \) (shape \( [N, \text{dim\_outputs}^{[l]}] \))

### Forward Propagation:

1. Input \( X \) (shape \( [N, D] \))
2. \( A^{[0]} = X \)
3. For each layer \( l \) from 1 to \( L \):
    - \( Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]} \) (Matrix multiplication and broadcasting for bias)
    - \( A^{[l]} = g^{[l]}(Z^{[l]}) \)

### Softmax Function for Output Layer \( L \):

\[ A^{[L]} = \text{Softmax}(Z^{[L]}) \]

### Loss Function (Cross-Entropy Loss for \( N \) samples):

\[ J = -\frac{1}{N} \sum_{n=1}^{N} \sum_{i=1}^{K} y_{n,i} \log(a^{[L]}_{n,i}) \]

### Backward Propagation:

1. Compute the error term for the output layer:
\[
\Delta^{[L]} = A^{[L]} - Y
\]
   \( Y \) is the label matrix of shape \( [N, K] \)

2. For each layer \( l \) from \( L-1 \) to 1:
    - \( \Delta^{[l]} = (\Delta^{[l+1]} (W^{[l+1]})^T) \odot g^{[l]'}(Z^{[l]}) \)

3. Compute gradients:
\[
\frac{\partial J}{\partial W^{[l]}} = \frac{1}{N} A^{[l-1]T} \Delta^{[l]}
\]
\[
\frac{\partial J}{\partial b^{[l]}} = \frac{1}{N} \sum_{n=1}^{N} \Delta^{[l]}_n
\]

### Update Parameters:

\[
W^{[l]} = W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}
\]
\[
b^{[l]} = b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}}
\]

In this formulation, each of the matrices and vectors are adapted to accommodate multiple samples, and the calculations are vectorized for efficiency.
# Mathematical foundations

Only the ideas used in the notebooks.

## Linear algebra
- A dataset is a matrix X with n rows and p columns. A linear model predicts with `X w`.
- Solving least squares: minimise `||y - Xw||^2`. The normal equation is `w = (X^T X)^-1 X^T y` when `X^T X` is invertible. Ridge adds `lambda I`, which makes it invertible.
- PCA uses the eigenvectors of the covariance matrix, or the singular value decomposition of centred X.

## Calculus and optimisation
- The gradient points in the direction of fastest increase. Gradient descent steps against it: `w <- w - eta * grad`.
- A step size `eta` that is too large diverges, one that is too small is slow. See `00_fundamentals/06_calculus_and_optimization`.
- Convex losses (squared error, logistic loss) have no local minima other than the global one. Neural network losses are not convex.

## Probability
- Bayes' rule: `P(A|B) = P(B|A) P(A) / P(B)`. Naive Bayes assumes features are independent given the class.
- Maximum likelihood: choose parameters that make the data most probable. Squared error is maximum likelihood under Gaussian noise.
- Expectation is a probability weighted average. Return in RL is `G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...`.

## Dynamic programming (RL)
The Bellman optimality equation `V*(s) = max_a sum_s' P(s'|s,a) [r + gamma V*(s')]` is solved by value iteration. The tabular notebook checks it against the known optimal success probability of the 4x4 slippery FrozenLake (about 0.8235 with gamma 1), which is also a unit test.

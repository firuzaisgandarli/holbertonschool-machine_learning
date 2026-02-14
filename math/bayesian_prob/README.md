# Bayesian Probability

This project focuses on implementing fundamental Bayesian probability concepts using Python and NumPy.

All files are written for:
- Ubuntu 20.04 LTS
- Python 3.9
- NumPy 1.25.2
- Pycodestyle 2.11.1

---

## Learning Objectives

By completing this project, we understand:

- What likelihood means in probability
- The Binomial distribution
- How Bayesian inference works
- How to compute probabilities using NumPy
- Proper input validation and error handling

---

## Tasks

### 0. Likelihood

We are given:

- `n`: total number of trials
- `x`: number of successes
- `P`: array of hypothetical probabilities

The goal is to compute the likelihood:

\[
L(P) = \binom{n}{x} P^x (1 - P)^{n-x}
\]

The function:

```python
def likelihood(x, n, P):

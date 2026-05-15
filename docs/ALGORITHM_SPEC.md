# Algorithm Specification — Bandit-STOR MVP

## 1. Logged Data

Each row contains:

```text
x_i       context
C_i       candidate set
a_i       logged action
r_i       observed reward
mu_i      logged propensity score for a_i
pos_i     optional position
```

## 2. Reward Model

Train:

```text
q_hat(x,a) = E[r | x,a]
```

Only the logged action receives a supervised reward label.

Unobserved actions are not converted into zero-reward labels.

## 3. Actor

The actor produces logits:

```text
z_theta(x, C) -> [K]
```

Policy:

```text
pi_theta(.|x,C) = sparsemax(z_theta(x,C) / temperature)
```

## 4. Tsallis q=2 Entropy

```text
H_2(pi) = 1 - sum_a pi(a)^2
```

## 5. Alpha Divergence

```text
D_alpha(p || q) = 1 / [alpha(alpha - 1)] * (sum_a p(a)^alpha q(a)^(1-alpha) - 1)
```

Default:

```text
alpha = 1.5
```

## 6. Doubly Robust Policy Value

```text
V_DR(pi) = mean_i [
  sum_a pi(a|x_i) q_hat(x_i,a)
  + pi(a_i|x_i) / mu_hat(a_i|x_i) * (r_i - q_hat(x_i,a_i))
]
```

## 7. Actor Loss

```text
L_actor =
  -V_DR(pi_theta)
  + beta_alpha * D_alpha(pi_theta || mu_hat)
  - lambda_tsallis * H_2(pi_theta)
  + lambda_support * unsupported_action_mass
```

## 8. Unsupported Action Mass

```text
unsupported_action_mass = mean_i sum_a pi_theta(a|x_i) * 1[mu_hat(a|x_i) < mu_min]
```

## 9. OPE Metrics

Compute:

```text
IPS = mean_i [pi(a_i|x_i) / mu(a_i|x_i) * r_i]
SNIPS = sum_i [w_i r_i] / sum_i [w_i]
DR = V_DR(pi)
ESS = (sum_i w_i)^2 / sum_i w_i^2
max_weight = max_i w_i
logged_action_coverage = mean_i 1[pi(a_i|x_i) > 0 and mu(a_i|x_i) > mu_min]
support_overlap = ESS / n under uniform logging, else logged_action_coverage
DRos(lambda) sensitivity over lambda in {0.1, 0.3, 1, 3, 10, 30, 100}
```

## 10. Numerical Safety

- Clamp `mu` using `mu_min`.
- Clip importance weights during training and report unclipped diagnostics separately.
- Alpha-divergence inputs `p` and `q` are clamped to avoid NaNs.
- Normalize probabilities after masks.

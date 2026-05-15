import pytest
import torch

from bandit_stor.objectives import (
    alpha_divergence,
    doubly_robust_value,
    effective_sample_size,
    importance_weights,
    ips_value,
    snips_value,
    sparse_tsallis_actor_loss,
    tsallis_q2_entropy,
)


def test_tsallis_entropy_known_values():
    deterministic = torch.tensor([[1.0, 0.0, 0.0]])
    uniform = torch.tensor([[1 / 4, 1 / 4, 1 / 4, 1 / 4]])
    assert torch.allclose(tsallis_q2_entropy(deterministic), torch.tensor([0.0]))
    assert torch.allclose(tsallis_q2_entropy(uniform), torch.tensor([0.75]))


def test_alpha_divergence_identical_zero_and_support_sensitive():
    p = torch.tensor([[0.2, 0.8]])
    assert alpha_divergence(p, p).item() == pytest.approx(0.0, abs=1e-5)
    q_good = torch.tensor([[0.2, 0.8]])
    q_bad = torch.tensor([[0.8, 0.2]])
    assert alpha_divergence(p, q_bad).item() > alpha_divergence(p, q_good).item()
    tiny = alpha_divergence(torch.tensor([[1.0, 0.0]]), torch.tensor([[1e-12, 1.0]]))
    assert torch.isfinite(tiny).all()


def test_ips_snips_ess_and_dr_toy_examples():
    reward = torch.tensor([1.0, 0.0])
    pi_logged = torch.tensor([0.5, 0.25])
    pscore = torch.tensor([0.25, 0.25])
    weights = importance_weights(pi_logged, pscore)
    assert torch.allclose(weights, torch.tensor([2.0, 1.0]))
    assert ips_value(weights, reward).item() == pytest.approx(1.0)
    assert snips_value(weights, reward).item() == pytest.approx(2 / 3)
    assert effective_sample_size(torch.ones(4)).item() == pytest.approx(4.0)

    pi = torch.tensor([[0.5, 0.5], [0.25, 0.75]])
    q_zero = torch.zeros_like(pi)
    logged = torch.tensor([0, 0])
    dr_as_ips = doubly_robust_value(pi, q_zero, logged, reward, pscore)
    assert dr_as_ips.item() == pytest.approx(ips_value(weights, reward).item())
    q_hat = torch.tensor([[0.4, 0.6], [0.1, 0.9]])
    reward_equal_q = torch.tensor([0.4, 0.1])
    dr_direct = doubly_robust_value(pi, q_hat, logged, reward_equal_q, pscore)
    assert dr_direct.item() == pytest.approx(float((pi * q_hat).sum(dim=1).mean()))


def test_actor_loss_is_finite_and_has_components():
    pi = torch.tensor([[0.7, 0.3], [0.2, 0.8]], requires_grad=True)
    behavior = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    q = torch.tensor([[0.5, 0.2], [0.1, 0.9]])
    out = sparse_tsallis_actor_loss(
        pi,
        behavior,
        q,
        torch.tensor([0, 1]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.5, 0.5]),
    )
    assert torch.isfinite(out.loss)
    assert {"loss", "dr_value", "scaled_dr_value", "reward_value_scale", "alpha_divergence", "tsallis_entropy", "unsupported_action_mass"} <= set(out.components)
    out.loss.backward()
    assert pi.grad is not None

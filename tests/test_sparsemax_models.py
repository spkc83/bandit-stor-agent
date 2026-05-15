import torch

from bandit_stor.models import BehaviorPolicyModel, RewardModel, SparseTsallisActor, sparsemax


def test_sparsemax_valid_sparse_masked_distribution_and_gradients():
    logits = torch.tensor([[3.0, 1.0, -2.0], [0.1, 0.2, 0.3]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    probs = sparsemax(logits, mask=mask)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2))
    assert (probs >= 0).all()
    assert probs[0, 2].item() == 0.0
    assert (probs == 0).any()
    probs[:, 0].sum().backward()
    assert logits.grad is not None


def test_models_emit_expected_shapes_and_policy_distribution():
    context = torch.randn(4, 2)
    action_context = torch.randn(4, 5, 1)
    mask = torch.ones(4, 5, dtype=torch.bool)
    behavior = BehaviorPolicyModel(2, 1)
    reward = RewardModel(2, 1)
    actor = SparseTsallisActor(2, 1)
    mu = behavior(context, action_context, mask)
    q = reward(context, action_context)
    pi = actor(context, action_context, mask)
    assert mu.shape == (4, 5)
    assert q.shape == (4, 5)
    assert pi.shape == (4, 5)
    assert torch.allclose(mu.sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(pi.sum(dim=1), torch.ones(4), atol=1e-6)

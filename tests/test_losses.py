import torch
import pytest

from src.training.losses import DiceLoss, BCEDiceLoss


# ── helpers ───────────────────────────────────────────────────────────

def _logits(prob: float, shape: tuple = (2, 1, 8, 8)) -> torch.Tensor:
    """Create a constant logit tensor that maps to *prob* after sigmoid."""
    return torch.full(shape, torch.logit(torch.tensor(prob)).item())


# ── DiceLoss ──────────────────────────────────────────────────────────

class TestDiceLoss:
    def test_perfect_prediction_loss_near_zero(self):
        """When prediction matches target perfectly, Dice ≈ 0."""
        targets = torch.ones(2, 1, 8, 8)
        logits = _logits(0.99)  # sigmoid → ~0.99
        loss = DiceLoss()(logits, targets)
        assert loss.item() < 0.05

    def test_worst_prediction_loss_near_one(self):
        """When prediction is opposite of target, Dice ≈ 1."""
        targets = torch.ones(2, 1, 8, 8)
        logits = _logits(0.01)  # sigmoid → ~0.01
        loss = DiceLoss()(logits, targets)
        assert loss.item() > 0.9

    def test_both_empty_loss_less_than_mismatched(self):
        """Empty pred + empty target should produce lower loss than empty pred + full target."""
        empty_targets = torch.zeros(2, 1, 8, 8)
        full_targets = torch.ones(2, 1, 8, 8)
        logits = _logits(0.01)  # sigmoid → ~0.01 (near-zero prediction)

        loss_both_empty = DiceLoss()(logits, empty_targets)
        loss_mismatched = DiceLoss()(logits, full_targets)

        assert loss_both_empty.item() < loss_mismatched.item()

    def test_output_is_scalar(self):
        loss = DiceLoss()(torch.randn(4, 1, 8, 8), torch.ones(4, 1, 8, 8))
        assert loss.dim() == 0

    def test_loss_in_valid_range(self):
        loss = DiceLoss()(torch.randn(4, 1, 16, 16), torch.ones(4, 1, 16, 16))
        assert 0.0 <= loss.item() <= 1.0

    def test_smooth_parameter(self):
        """Different smooth values should produce different losses for same input."""
        targets = torch.zeros(2, 1, 8, 8)
        logits = torch.zeros(2, 1, 8, 8)  # sigmoid → 0.5
        loss_s1 = DiceLoss(smooth=1.0)(logits, targets).item()
        loss_s100 = DiceLoss(smooth=100.0)(logits, targets).item()
        assert loss_s1 != pytest.approx(loss_s100, abs=1e-4)

    def test_supports_3d_input(self):
        """Should work with (B, H, W) tensors without channel dim."""
        targets = torch.ones(2, 8, 8)
        logits = _logits(0.99, shape=(2, 8, 8))
        loss = DiceLoss()(logits, targets)
        assert loss.item() < 0.05

    def test_gradient_flows(self):
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        targets = torch.ones(2, 1, 8, 8)
        loss = DiceLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)


# ── BCEDiceLoss ───────────────────────────────────────────────────────

class TestBCEDiceLoss:
    def test_perfect_prediction_low_loss(self):
        targets = torch.ones(2, 1, 8, 8)
        logits = _logits(0.99)
        loss = BCEDiceLoss()(logits, targets)
        assert loss.item() < 0.1

    def test_output_is_scalar(self):
        loss = BCEDiceLoss()(torch.randn(4, 1, 8, 8), torch.ones(4, 1, 8, 8))
        assert loss.dim() == 0

    def test_loss_non_negative(self):
        loss = BCEDiceLoss()(torch.randn(4, 1, 16, 16), torch.ones(4, 1, 16, 16))
        assert loss.item() >= 0.0

    def test_gradient_flows(self):
        logits = torch.randn(2, 1, 8, 8, requires_grad=True)
        targets = torch.ones(2, 1, 8, 8)
        loss = BCEDiceLoss()(logits, targets)
        loss.backward()
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)

    def test_weights_affect_loss(self):
        """Changing bce/dice weights should change the total loss."""
        logits = torch.randn(2, 1, 8, 8)
        targets = torch.ones(2, 1, 8, 8)

        loss_equal = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)(logits, targets)
        loss_bce_heavy = BCEDiceLoss(bce_weight=5.0, dice_weight=0.1)(logits, targets)
        loss_dice_heavy = BCEDiceLoss(bce_weight=0.1, dice_weight=5.0)(logits, targets)

        # All three should be different
        assert loss_equal.item() != pytest.approx(loss_bce_heavy.item(), abs=1e-4)
        assert loss_equal.item() != pytest.approx(loss_dice_heavy.item(), abs=1e-4)

    def test_zero_bce_weight_equals_dice_only(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 1, 8, 8)
        targets = (torch.rand(4, 1, 8, 8) > 0.5).float()

        combined = BCEDiceLoss(bce_weight=0.0, dice_weight=1.0)(logits, targets)
        dice_only = DiceLoss()(logits, targets)

        assert combined.item() == pytest.approx(dice_only.item(), abs=1e-6)

    def test_zero_dice_weight_equals_bce_only(self):
        torch.manual_seed(0)
        logits = torch.randn(4, 1, 8, 8)
        targets = (torch.rand(4, 1, 8, 8) > 0.5).float()

        combined = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)(logits, targets)
        bce_only = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, targets
        )

        assert combined.item() == pytest.approx(bce_only.item(), abs=1e-6)

    def test_pos_weight(self):
        """pos_weight should increase loss for false negatives."""
        logits = _logits(0.3)  # prediction ≈ 0.3, target = 1 → mostly FN
        targets = torch.ones(2, 1, 8, 8)

        loss_no_pw = BCEDiceLoss(pos_weight=None)(logits, targets)
        loss_high_pw = BCEDiceLoss(pos_weight=5.0)(logits, targets)

        assert loss_high_pw.item() > loss_no_pw.item()

    def test_default_weights_are_one(self):
        criterion = BCEDiceLoss()
        assert criterion.bce_weight == 1.0
        assert criterion.dice_weight == 1.0

    def test_supports_3d_input(self):
        targets = torch.ones(2, 8, 8)
        logits = _logits(0.99, shape=(2, 8, 8))
        loss = BCEDiceLoss()(logits, targets)
        assert loss.item() < 0.1

    def test_batch_size_one(self):
        logits = torch.randn(1, 1, 32, 32)
        targets = (torch.rand(1, 1, 32, 32) > 0.5).float()
        loss = BCEDiceLoss()(logits, targets)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

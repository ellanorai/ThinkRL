from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from thinkrl.models.utils import (
    EMAModel,
    count_parameters,
    create_reference_model,
    disable_gradient_checkpointing,
    enable_gradient_checkpointing,
    freeze_layers,
    freeze_model,
    get_model_device,
    get_model_dtype,
    model_memory_footprint,
    share_model_weights,
    unfreeze_model,
    update_reference_model,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 5)
        self.embed = nn.Embedding(10, 10)


class TestReferenceModel:
    def test_create_reference_model_copy(self):
        model = SimpleModel()
        ref_model = create_reference_model(model, share_weights=False)
        assert ref_model is not model
        assert not next(ref_model.parameters()).requires_grad

        # Verify values match initially
        assert torch.equal(ref_model.layer1.weight, model.layer1.weight)

        # Modify original, ref should not change
        with torch.no_grad():
            model.layer1.weight.add_(1.0)
        assert not torch.equal(ref_model.layer1.weight, model.layer1.weight)

    def test_create_reference_model_share(self):
        model = SimpleModel()
        ref_model = create_reference_model(model, share_weights=True)
        assert ref_model is model
        # Should be frozen now
        assert not next(ref_model.parameters()).requires_grad

    def test_update_reference_model(self):
        model = SimpleModel()
        ref_model = create_reference_model(model, share_weights=False)

        # Change model weights
        with torch.no_grad():
            model.layer1.weight.fill_(1.0)
            ref_model.layer1.weight.fill_(0.0)

        # Update with tau=1.0 (full copy)
        update_reference_model(ref_model, model, tau=1.0)
        assert torch.equal(ref_model.layer1.weight, model.layer1.weight)

        # Update with tau=0.5
        with torch.no_grad():
            model.layer1.weight.fill_(2.0)
        # ref is 1.0 (from previous step)
        # new ref = 0.5 * 1.0 + 0.5 * 2.0 = 1.5
        update_reference_model(ref_model, model, tau=0.5)
        assert torch.allclose(ref_model.layer1.weight, torch.tensor(1.5))

    def test_share_model_weights(self):
        source = SimpleModel()
        target = SimpleModel()

        # Make them different
        with torch.no_grad():
            source.layer1.weight.fill_(1.0)
            target.layer1.weight.fill_(2.0)

        share_model_weights(source, target)

        # After sharing, values should be same
        assert torch.all(target.layer1.weight == source.layer1.weight)

        # Check if they are the same memory object (storage)
        # load_state_dict by default COPIES, so they should NOT be same object
        # and checking data pointers or just checking modification behavior

        with torch.no_grad():
            source.layer1.weight.fill_(3.0)

        # Target should retain 1.0 (since it was a copy)
        # So they should differ now
        assert torch.all(target.layer1.weight == 1.0)
        assert torch.all(source.layer1.weight == 3.0)


class TestFreezing:
    def test_freeze_unfreeze(self):
        model = SimpleModel()
        freeze_model(model)
        for p in model.parameters():
            assert not p.requires_grad
        assert not model.training

        unfreeze_model(model)
        for p in model.parameters():
            assert p.requires_grad
        assert model.training

    def test_freeze_layers_by_name(self):
        model = SimpleModel()
        freeze_layers(model, layer_names=["layer1"])
        assert not model.layer1.weight.requires_grad
        assert model.layer2.weight.requires_grad

    def test_freeze_layers_num(self):
        # Need a model structure that matches regex in utils.py
        # regex: (?:layer|block)[._]?(\d+)

        class SequentialModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(3)])

        model = SequentialModel()
        # names will be layers.0.weight, layers.1.weight...
        # regex expects "layer" or "block" in name. "layers" contains "layer".
        # "layers.0" -> match group 1 is "0".

        freeze_layers(model, num_layers=2)

        assert not model.layers[0].weight.requires_grad
        assert not model.layers[1].weight.requires_grad
        assert model.layers[2].weight.requires_grad

    def test_freeze_embeddings(self):
        model = SimpleModel()
        freeze_layers(model, freeze_embeddings=True)
        assert not model.embed.weight.requires_grad
        # "layer" params should still be trainable (default)
        assert model.layer1.weight.requires_grad


class TestEMAModel:
    def test_ema_update(self):
        model = SimpleModel()
        ema = EMAModel(model, decay=0.5, update_every=1)

        # Initial shadow params match model
        assert torch.equal(ema.shadow_params["layer1.weight"], model.layer1.weight)

        # Change model
        with torch.no_grad():
            model.layer1.weight.fill_(1.0)  # Model became 1.0
            # Shadow was random (say 0.0 for simplicity of thought, but it's random).
            # Let's force values to be deterministic
            ema.shadow_params["layer1.weight"].fill_(0.0)

        ema.update()
        # new shadow = 0.5 * 0.0 + (1-0.5) * 1.0 = 0.5
        assert torch.allclose(ema.shadow_params["layer1.weight"], torch.tensor(0.5))

    def test_ema_context_manager(self):
        model = SimpleModel()
        # Set model weights
        with torch.no_grad():
            model.layer1.weight.fill_(1.0)

        # Initialize EMA, it copies model weights (1.0)
        ema = EMAModel(model)

        # Set EMA weights to 2.0 manually
        with torch.no_grad():
            ema.shadow_params["layer1.weight"].fill_(2.0)

        with ema.average_parameters():
            # Inside context, model should have EMA weights (2.0)
            # Use allclose to handle scalar comparison with tensor
            assert torch.all(model.layer1.weight == 2.0)

        # Outside, model should have original weights (1.0)
        assert torch.all(model.layer1.weight == 1.0)

    def test_state_dict(self):
        model = SimpleModel()
        ema = EMAModel(model)
        state = ema.state_dict()
        assert "shadow_params" in state
        assert "step" in state

        ema2 = EMAModel(model)
        ema2.load_state_dict(state)
        assert ema2.step == ema.step


class TestAnalysis:
    def test_count_parameters(self):
        model = SimpleModel()
        stats = count_parameters(model)
        assert "trainable" in stats
        assert "frozen" in stats
        assert "total" in stats
        assert stats["total"] > 0

    def test_get_model_device(self):
        model = SimpleModel()
        device = get_model_device(model)
        assert isinstance(device, torch.device)

    def test_get_model_dtype(self):
        model = SimpleModel()
        dtype = get_model_dtype(model)
        assert dtype == torch.float32

    def test_memory_footprint(self):
        model = SimpleModel()
        mem = model_memory_footprint(model)
        assert "parameters_mb" in mem
        assert mem["total_mb"] > 0

    def test_gradient_checkpointing(self):
        model = MagicMock()
        model.gradient_checkpointing_enable = MagicMock()
        assert enable_gradient_checkpointing(model)
        model.gradient_checkpointing_enable.assert_called()

        model.gradient_checkpointing_disable = MagicMock()
        assert disable_gradient_checkpointing(model)
        model.gradient_checkpointing_disable.assert_called()

    def test_gradient_checkpointing_unsupported(self):
        model = SimpleModel()  # Linear layers usually don't have this method at module level unless HF
        # nn.Module doesn't have gradient_checkpointing_enable by default
        assert not enable_gradient_checkpointing(model)
        assert not disable_gradient_checkpointing(model)

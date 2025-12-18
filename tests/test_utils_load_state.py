import pytest

torch = pytest.importorskip("torch")

from utils.utils import load_state_dict_flexible


class TinyModel(torch.nn.Module):
    def __init__(self, include_plip_head=False):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        if include_plip_head:
            self.plip_head = torch.nn.Linear(2, 2)


def test_strips_module_prefix_and_logs(tmp_path):
    model = TinyModel(include_plip_head=False)
    checkpoint_model = TinyModel(include_plip_head=False)
    checkpoint_state = {
        f'module.{k}': v.clone()
        for k, v in checkpoint_model.state_dict().items()
    }

    log_path = tmp_path / "load_state.log"
    missing, dropped, unexpected = load_state_dict_flexible(
        model, checkpoint_state, strict=False, log_path=log_path
    )

    assert missing == []
    assert dropped == []
    assert unexpected == []
    assert torch.equal(model.backbone.weight, checkpoint_model.backbone.weight)
    assert "module.backbone.weight" in log_path.read_text()


def test_ignores_plip_head_when_features_disabled(tmp_path):
    model = TinyModel(include_plip_head=False)
    checkpoint_model = TinyModel(include_plip_head=True)
    checkpoint_state = checkpoint_model.state_dict()

    log_path = tmp_path / "load_state_plip.log"
    missing, dropped, unexpected = load_state_dict_flexible(
        model, checkpoint_state, strict=False, log_path=log_path
    )

    assert missing == []
    assert unexpected == []
    assert set(dropped) == {"plip_head.weight", "plip_head.bias"}
    assert torch.equal(model.backbone.bias, checkpoint_model.backbone.bias)
    log_text = log_path.read_text()
    assert "plip_head.weight" in log_text
    assert "EMA will track them from initialization" in log_text

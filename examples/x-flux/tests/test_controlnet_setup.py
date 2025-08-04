import pytest
from src.flux.xflux_pipeline import XFluxPipeline

def test_set_controlnet(monkeypatch):
    pipe = XFluxPipeline("flux-dev", "cpu", False)
    # Mock controlnet and checkpoint loading
    monkeypatch.setattr("src.flux.util.load_controlnet", lambda model_type: type("Dummy", (), {"to": lambda self, dtype: self, "load_state_dict": lambda self, ckpt, strict=False: None})())
    monkeypatch.setattr("src.flux.util.load_checkpoint", lambda local_path, repo_id, name: {})
    monkeypatch.setattr("src.flux.util.Annotator", lambda control_type: lambda img, w, h: img)
    pipe.set_controlnet("canny", "dummy.ckpt", "XLabs-AI/flux-controlnet-canny-v3", "flux-canny-controlnet-v3.safetensors")
    assert pipe.controlnet_loaded
    assert pipe.control_type == "canny"

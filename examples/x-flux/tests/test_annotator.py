import pytest
from src.flux.xflux_pipeline import XFluxPipeline

def test_annotator(monkeypatch):
    pipe = XFluxPipeline("flux-dev", "cpu", False)
    # Setup controlnet
    monkeypatch.setattr("src.flux.util.load_controlnet", lambda model_type: type("Dummy", (), {"to": lambda self, dtype: self, "load_state_dict": lambda self, ckpt, strict=False: None})())
    monkeypatch.setattr("src.flux.util.load_checkpoint", lambda local_path, repo_id, name: {})
    monkeypatch.setattr("src.flux.util.Annotator", lambda control_type: lambda img, w, h: "annotated_img")
    pipe.set_controlnet("canny")
    # Simulate annotator call
    result = pipe.annotator("input_canny1.png", 1024, 1024)
    assert result == "annotated_img"

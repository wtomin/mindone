import pytest
from src.flux.xflux_pipeline import XFluxPipeline

def test_pipeline_init(monkeypatch):
    monkeypatch.setattr("src.flux.util.load_clip", lambda: "clip")
    monkeypatch.setattr("src.flux.util.load_t5", lambda max_length=512: "t5")
    monkeypatch.setattr("src.flux.util.load_ae", lambda model_type: "ae")
    monkeypatch.setattr("src.flux.util.load_flow_model", lambda model_type: "model")
    pipe = XFluxPipeline("flux-dev", "cpu", False)
    assert pipe.model_type == "flux-dev"
    assert pipe.device.type == "cpu"

def test_forward(monkeypatch):
    pipe = XFluxPipeline("flux-dev", "cpu", False)
    monkeypatch.setattr(pipe, "forward", lambda *a, **kw: "output_img")
    result = pipe.construct(
        prompt="cyberpank dining room, full hd, cinematic",
        width=1024,
        height=1024,
        guidance=4,
        num_steps=25,
        seed=123456789,
        controlnet_image="annotated_img",
        timestep_to_start_cfg=1,
        true_gs=4,
        control_weight=0.8,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    )
    assert result == "output_img"

def test_pipeline_call(monkeypatch):
    pipe = XFluxPipeline("flux-dev", "cpu", False)
    pipe.controlnet_loaded = True
    pipe.annotator = lambda img, w, h: "annotated_img"
    monkeypatch.setattr(pipe, "forward", lambda *a, **kw: "output_img")
    result = pipe(
        prompt="cyberpank dining room, full hd, cinematic",
        controlnet_image="input_canny1.png",
        width=1024,
        height=1024,
        guidance=4,
        num_steps=25,
        seed=123456789,
        true_gs=4,
        control_weight=0.8,
        neg_prompt="",
        timestep_to_start_cfg=1,
        image_prompt=None,
        neg_image_prompt=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    )
    assert result == "output_img"

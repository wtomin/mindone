"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/qwenimage/__init__.py."""

from typing import TYPE_CHECKING

from ...utils import _LazyModule

_dummy_objects = {}
_import_structure = {}

_import_structure["pipeline_qwenimage"] = ["QwenImagePipeline"]

if TYPE_CHECKING:
    from .pipeline_qwenimage import QwenImagePipeline
    from .pipeline_qwenimage_img2img import QwenImageImg2ImgPipeline
    from .pipeline_qwenimage_inpaint import QwenImageInpaintPipeline

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
    for name, value in _dummy_objects.items():
        setattr(sys.modules[__name__], name, value)

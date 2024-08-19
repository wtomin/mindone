from huggingface_hub import snapshot_download

from .image_detokenize import LaVITDetokenizer
from .lavit_for_generation import LaVITforGeneration
from .lavit_for_understanding import LaVITforUnderstanding
from .transform import LaVITImageProcessor, LaVITQuestionProcessor


# Building the Model
def build_model(
    model_path="./",
    model_dtype="bf16",
    amp_level="O2",
    use_flash_attention=False,
    understanding=True,
    load_tokenizer=True,
    pixel_decoding="highres",
    check_safety=True,
    local_files_only=False,
    model_sub_dir="language_model",
):
    """
    model_path (str): The local directory for the saving the model weight
    model_dtype (str): The precision dtype of the model in inference, bf16 or fp16
    use_flash_attention (bool): default=False, If set True, use flash attention to save the memory in the eva clip
    understanding (bool): If set True, use LaVIT for multi-modal understanding, else used for generation
    load_tokenizer (bool): Whether to load the tokenizer encoder during the image generation. For text-to-image generation,
        The visual tokenizer is not needed, set it to `False` for saving the memory. When using for the
        multi-modal synthesis (the input image needs to be tokenizd to dircrete ids), the load_tokenizer must be set to True.
    pixel_decoding (str): [highres | lowres]: default is `highres`: using the high resolution decoding
        for generating high-quality images, if set to `lowres`, using the origin decoder to generate 512 x 512 image
    check_safety (bool): Should be set to True to enable the image generation safety check
    local_files_only (bool): If you have already downloaded the LaVIT checkpoint to the model_path,
    set the local_files_only=True to avoid loading from remote
    """

    if not local_files_only:
        print("Downloading the LaVIT checkpoint from huggingface")
        snapshot_download(
            "rain1011/LaVIT-7B-v2",
            local_dir=model_path,
            local_files_only=local_files_only,
            local_dir_use_symlinks=False,
        )

    if understanding:
        lavit = LaVITforUnderstanding(
            model_path=model_path,
            model_dtype=model_dtype,
            amp_level=amp_level,
            use_flash_attention=use_flash_attention,
            model_sub_dir=model_sub_dir,
        )
    else:
        lavit = LaVITforGeneration(
            model_path=model_path,
            model_dtype=model_dtype,
            amp_level=amp_level,
            use_flash_attention=use_flash_attention,
            check_safety=check_safety,
            load_tokenizer=load_tokenizer,
            pixel_decoding=pixel_decoding,
            model_sub_dir=model_sub_dir,
        )

    return lavit

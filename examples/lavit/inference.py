import argparse
import logging
import os
import sys

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))

from models import build_model
from utils import init_env

logger = logging.getLogger(__name__)


def example_for_understanding(model_path, model_dtype, load_from_local=False):
    # Building model and load weight
    model = build_model(
        model_path=model_path, model_dtype=model_dtype, understanding=True, local_files_only=load_from_local
    )

    # Image Captioning
    image_path = "demo/caption_image.jpg"
    caption = model.generate({"image": image_path})[0]
    print(caption)

    # Visual Question Answering
    image_path = "demo/qa_image.jpg"
    question = "What's that drink in the glass?"
    print("Question:", question)
    answer = model.predict_answers({"image": image_path, "text_input": question}, max_len=10)[0]
    print("The answer is: ", answer)


def example_for_generation(model_path, model_dtype, load_from_local=False):
    # Building model and load weight
    model = build_model(
        model_path=model_path,
        model_dtype=model_dtype,
        check_safety=False,
        understanding=False,
        local_files_only=load_from_local,
    )

    # LaVIT support 6 different image aspect ratios
    ratio_dict = {
        "1:1": (1024, 1024),
        "4:3": (896, 1152),
        "3:2": (832, 1216),
        "16:9": (768, 1344),
        "2:3": (1216, 832),
        "3:4": (1152, 896),
    }

    # The image aspect ratio you want to generate
    ratio = "1:1"
    height, width = ratio_dict[ratio]

    # Text-to-Image Generation
    prompt = "A photo of an astronaut riding a horse in the forest."
    image = model.generate_image(prompt, width=width, height=height, guidance_scale_for_llm=4.0, num_return_images=1)[0]
    image.save("output/t2i_output.jpg")

    # Multi-modal Image synthesis
    image_prompt = "demo/dog.jpg"
    text_prompt = "It is running in the snow"
    input_prompts = [(image_prompt, "image"), (text_prompt, "text")]
    image = model.multimodal_synthesis(
        input_prompts, width=width, height=height, guidance_scale_for_llm=5.0, num_return_images=1
    )[0]
    image.save("output/it2i_output.jpg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # MS new args
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument(
        "--model_path",
        type=str,
        default="LaVIT_checkpoint",
        help="The directory to the checkpoint, will download from huggingface if not existent.",
    )
    parser.add_argument(
        "--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"], help="The precision dtype"
    )
    parser.add_argument("--seed", type=int, default=1234, help="The random seed")
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    args = parser.parse_args()

    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        device_target=args.device,
        jit_level=args.jit_level,
    )
    model_path = args.model_path
    load_from_local = os.path.exists(args.model_path) and not len(os.listdir(args.model_path)) == 0

    # For Multi-Modal Understanding
    example_for_understanding(model_path, args.precision, load_from_local=load_from_local)

    # For Multi-Modal Generation
    example_for_generation(model_path, args.precision, load_from_local=load_from_local)

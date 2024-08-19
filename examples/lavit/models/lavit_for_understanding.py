import logging
import os

import cv2
from mindformers.models.llama import LlamaForCausalLM, LlamaTokenizer
from models.modeling_visual_tokenizer import build_dynamic_tokenizer
from models.transform import LaVITImageProcessor
from utils import get_amp_model

import mindspore as ms
from mindspore import nn, ops


class LaVITforUnderstanding(nn.Cell):
    """
    The LaVIT Model for Multi-modal Understanding,
    this file is used for reading image contents and answering the questions.
    """

    def __init__(
        self,
        img_size=224,
        model_path="",
        model_dtype="bf16",
        amp_level="O2",
        apply_lemmatizer=True,
        use_flash_attention=True,
        model_sub_dir="language_model",
    ):
        """
        img_size: The input image size, should be 224 * 224
        model_path: The pre-trained model checkpoint path, the local path for downloaded LaVIT weight
        model_dtype: The precision of model weight during inference, should be set bf16 or fp16, default is bf16.
        apply_lemmatizer: when set to True, postprocess predict_answers() result with lemmas
        """
        super().__init__()
        assert img_size == 224, "Input Image Size should be set to 224"

        visual_vocab_size = 16384  # The visual vocab size of LaVIT is 16384
        print(f"Loading LaVIT Model Weight from {model_path}, model precision: {model_dtype}")

        self.llama_tokenizer = LlamaTokenizer.from_pretrained(os.path.join(model_path, model_sub_dir))
        self.llama_model = LlamaForCausalLM.from_pretrained(
            os.path.join(model_path, model_sub_dir),
        )
        for param in self.llama_model.get_parameters():
            param.requires_grad = False
        self.llama_model = get_amp_model(self.llama_model, self.dtype, amp_level)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.visual_vocab_size = visual_vocab_size
        print(f"The Visual Vocab Size is {self.visual_vocab_size}")
        print(f"The llama tokenizer vocab size is {len(self.llama_tokenizer)}")

        self.visual_tokenizer = build_dynamic_tokenizer(model_path, for_understanding=True, model_sub_dir=model_sub_dir)
        self.model_dtype = model_dtype
        self.apply_lemmatizer = apply_lemmatizer
        self._lemmatizer = None
        self.processer = LaVITImageProcessor(image_size=img_size)

    @property
    def dtype(self):
        if self.model_dtype == "fp16":
            dtype = ms.float16
        elif self.model_dtype == "bf16":
            dtype = ms.bfloat16
        elif self.model_dtype == "fp32":
            # The default dtype is fp16
            dtype = ms.float32
        else:
            raise NotImplementedError
        return dtype

    def process_image(self, image_inputs):
        if isinstance(image_inputs, ms.Tensor):
            assert len(image_inputs.shape) == 4, "Image Tensors should have shape (batch_size, 3, H, W)"
            return image_inputs

        if not isinstance(image_inputs, list):
            assert isinstance(image_inputs, str)
            image_inputs = [image_inputs]

        image_tensors = []
        for image_path in image_inputs:
            # image = Image.open(image_path).convert('RGB')
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image = self.processer(image)
            image_tensors.append(image)

        image_tensors = ops.stack(image_tensors, axis=0)
        image_tensors = image_tensors
        return image_tensors

    def compute_dynamic_visual_embeds(self, image):
        image_embeds_list = self.visual_tokenizer.encode_features(image)
        batch_size = len(image_embeds_list)
        # Pad the image start and end tokens
        image_pad_token = ms.Tensor([32000, 32001], dtype=ms.int32)
        image_pad_embeds = self.llama_model.get_input_embeddings()(image_pad_token)  # [2, embed_dim]
        max_token_num = -1

        for i_b in range(batch_size):
            image_embeds_list[i_b] = ops.cat(
                [image_pad_embeds[:1], image_embeds_list[i_b], image_pad_embeds[1:]], axis=0
            )
            max_token_num = max(max_token_num, len(image_embeds_list[i_b]))

        # Pad with eos embeddings
        eos_id = self.llama_tokenizer.eos_token_id
        eos_id = ms.Tensor([eos_id], dtype=ms.int32)
        eos_embeds = self.llama_model.get_input_embeddings()(eos_id).unsqueeze(0)  # [1, 1, embed_dim]

        image_attns = ops.zeros((batch_size, max_token_num), dtype=ms.int32)
        image_embeds = eos_embeds.repeat_interleave(batch_size, axis=0).repeat_interleave(max_token_num, axis=1)

        # Use the left padding
        for i_b in range(batch_size):
            image_attns[i_b, -len(image_embeds_list[i_b]) :] = 1
            image_embeds[i_b, -len(image_embeds_list[i_b]) :] = image_embeds_list[i_b]

        return image_embeds, image_attns

    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=2,
        max_length=36,
        min_length=8,
        top_p=1.0,
        top_k=50,
        repetition_penalty=1,
        length_penalty=1,
        num_captions=1,
        temperature=1,
        **kwargs,
    ):
        """
        Usage:
            Generate the textual caption of input images
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (ms.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """

        image = self.process_image(samples["image"])

        if "prompt" in samples.keys():
            prompt = samples["prompt"]
        else:
            prompt = ""

        # Prepare image token ids
        image_embeds, image_attns = self.compute_dynamic_visual_embeds(image)

        if prompt != "":
            if isinstance(prompt, str):
                prompt = [prompt] * image.size(0)
            else:
                assert len(prompt) == image.size(0), "The number of prompts must be equal to the batch size."

            self.llama_tokenizer.padding_side = "left"
            prompt_tokens = self.llama_tokenizer(
                prompt, padding="longest", return_tensors=None, add_special_tokens=False
            )

            input_ids = ms.Tensor(prompt_tokens.input_ids)
            prompt_embeds = self.llama_model.get_input_embeddings()(input_ids)
            inputs_embeds = ops.cat([image_embeds, prompt_embeds], axis=1)
            attention_mask = ms.Tensor(prompt_tokens.attention_mask)
            attention_mask = ops.cat(
                [
                    image_attns,
                ],
                axis=1,
            )

        else:
            inputs_embeds = image_embeds
            attention_mask = image_attns

        # For captioning, supress the token ids > 32000 (Visual Tokens)
        supress_range = 32000 + self.visual_vocab_size + 2
        suppress_tokens = [x for x in range(32000, supress_range)]

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_length,
            min_new_tokens=min_length,
            suppress_tokens=suppress_tokens,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
        )

        output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        output_text = [text.split(".")[0] for text in output_text]
        return output_text

    def pad_input_embeds(self, image_embeds, image_attns, prompt_embeds, prompt_attns):
        # Concat the image and text embeddings
        batch_size = len(image_embeds)
        input_embeds, attention_mask = [], []

        for i_b in range(batch_size):
            image_embed = image_embeds[i_b]
            image_attn = image_attns[i_b]
            prompt_embed = prompt_embeds[i_b]  # [seq_len, embed_dim]
            prompt_attn = prompt_attns[i_b]  # [seq_len]

            prompt_len = int(prompt_attn.sum())
            pad_prompt_len = len(prompt_attn) - prompt_len

            if pad_prompt_len == 0:
                input_embed = ops.cat([image_embed, prompt_embed], axis=0)
                input_attn = ops.cat([image_attn, prompt_attn], axis=0)
            else:
                assert prompt_attn[:pad_prompt_len].sum() == 0
                input_embed = ops.cat([prompt_embed[:pad_prompt_len], image_embed, prompt_embed[-prompt_len:]], axis=0)
                input_attn = ops.cat([prompt_attn[:pad_prompt_len], image_attn, prompt_attn[-prompt_len:]], axis=0)

            input_embeds.append(input_embed)
            attention_mask.append(input_attn)

        input_embeds = ops.stack(input_embeds, axis=0)
        attention_mask = ops.stack(attention_mask, axis=0)

        return input_embeds, attention_mask

    def predict_answers(
        self,
        samples,
        num_beams=5,
        max_len=10,
        min_len=2,
        prompt="Question: {} Answer:",
        temperature=1,
        top_p=1.0,
        top_k=50,
        use_nucleus_sampling=False,
        length_penalty=0,
        **kwargs,
    ):
        """
        Usage:
            Answering the visual questions
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (ms.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
        Returns:
            answers (list): A list of strings of length batch_size.
        """
        image = self.process_image(samples["image"])

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if prompt:
            text_input = [prompt.format(question) for question in samples["text_input"]]
        else:
            text_input = samples["text_input"]

        self.llama_tokenizer.padding_side = "left"
        prompt_tokens = self.llama_tokenizer(
            text_input, padding="longest", return_tensors=None, add_special_tokens=False
        )

        prompt_embeds = self.llama_model.get_input_embeddings()(ms.Tensor(prompt_tokens.input_ids))
        image_embeds, image_attns = self.compute_dynamic_visual_embeds(image)

        # Concat the image and text embeddings to form left padding
        inputs_embeds, attention_mask = self.pad_input_embeds(
            image_embeds, image_attns, prompt_embeds, ms.Tensor(prompt_tokens.attention_mask)
        )

        supress_range = 32000 + self.visual_vocab_size + 2
        suppress_tokens = [x for x in range(32000, supress_range)]

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=use_nucleus_sampling,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_len,
            min_new_tokens=min_len,
            suppress_tokens=suppress_tokens,
            bos_token_id=self.llama_tokenizer.bos_token_id,
            eos_token_id=self.llama_tokenizer.eos_token_id,
            pad_token_id=self.llama_tokenizer.pad_token_id,
            length_penalty=length_penalty,
            early_stopping=True,
        )

        # print("output: ", outputs)
        output_text = self.llama_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        # The post posting for evaluation
        output_text = [text.split("\n")[0] for text in output_text]
        output_text = [text.split("question:")[0] for text in output_text]
        output_text = [text.split("Long answer:")[0] for text in output_text]
        output_text = [text.split(",")[0] for text in output_text]
        output_text = [text.split(".")[0] for text in output_text]

        # lemmatize the output
        output_text = self._lemmatize(output_text)

        return output_text

    def _lemmatize(self, answers):
        def apply(answer):
            answer = answer.lower()
            doc = self.lemmatizer(answer)

            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"]:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            answer = " ".join(words)

            return answer

        return [apply(answer) for answer in answers]

    @property
    def lemmatizer(self):
        if self._lemmatizer is None:
            try:
                import spacy

                self._lemmatizer = spacy.load("en_core_web_sm")
            except ImportError:
                logging.error(
                    """
                    Please install spacy and en_core_web_sm model to apply lemmatization.
                    python -m spacy download en_core_web_sm
                    OR
                    import spacy.cli
                    spacy.cli.download("en_core_web_sm")
                    """
                )
                exit(1)

        return self._lemmatizer

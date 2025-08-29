import unsloth
import torch

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from transformers.generation.streamers import TextStreamer

from subliminal_learning.llm import LLMConfig, ChatResponse, ChatMessage, LLMService, SamplingParams

torch.set_float32_matmul_precision('high')

'''
UNSLOTH INFERENCE SERVICE

Uses Unsloth engine for model loading and inference

'''


def get_unsloth_id(model_id: str):
    '''Convert to using unsloth model from hf model if available'''
    return {
        "google/gemma-2-2b-it": "unsloth/gemma-2-2b-it",
        "google/gemma-2-9b-it": "unsloth/gemma-2-9b-it",

    }.get(model_id, model_id)


class UnslothService(LLMService):
    def __init__(
            self, 
            llm_config: LLMConfig,
            debug: bool = False
        ):
        super().__init__(
            name = 'unsloth',
            llm_config = llm_config,
            supports_lora = True,  # Unsloth supports LoRA adapters
            supports_system_prompt = True,
            supports_steering = False,
            debug = debug
        )


    @classmethod
    def _get_model_tokenizer(cls, llm_config: LLMConfig, **kwargs):
        return FastLanguageModel.from_pretrained(
            model_name = get_unsloth_id(model_id = llm_config.base_model_id),
            attn_implementation = "flash_attention_2",
            load_in_4bit = llm_config.load_in_4bit,
            load_in_8bit = llm_config.load_in_8bit,
            **kwargs
        )

    @classmethod
    def _get_peft_model(cls, model: FastLanguageModel, **kwargs):
        return FastLanguageModel.get_peft_model(
            model,
            **kwargs
        )

    @classmethod
    def save_merged(cls, llm_config: LLMConfig, output_path: str):

        model, tokenizer = cls._get_model_tokenizer(
            llm_config = llm_config
        )
        model.save_pretrained_merged(output_path, tokenizer, save_method = "merged_16bit")


    # FIXME: Not checked if this works
    def chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None) -> list[ChatResponse]:

        if sampling_params is None:
            sampling_params = SamplingParams()
        
        # Load model (potentially from cache)
        if (self.model is None) or (self.tokenizer is None):
            self.get_model_tokenizer()
        assert(self.model is not None) and (self.tokenizer is not None), f"Experienced error loading model or tokenizer"

        # Add system prompts
        message_set = self.add_system_prompt(message_set)
        
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template = "gemma-2",
        )
            
        FastLanguageModel.for_inference(self.model)
        text_streamer = TextStreamer(self.tokenizer)
        
        # Convert sampling params to dict for compatibility
        sampling_dict = {
            'max_new_tokens': getattr(sampling_params, 'max_new_tokens', 512),
            'temperature': sampling_params.temperature
        }

        inputs = self.tokenizer.apply_chat_template(
            message_set, tokenize = True, add_generation_prompt = True, return_tensors = "pt"
        )

        output = self.model.generate(
            input_ids = inputs, 
            **sampling_params.model_dump(),
            use_cache = True
        )

        output_text = self.tokenizer.decode(output)

        return [ChatResponse(
            text = output_text,
            token_ids = output,
            prompt_token_ids = inputs
        )]


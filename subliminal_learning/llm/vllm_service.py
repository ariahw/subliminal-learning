import gc
import torch
import traceback
from typing import Any

from vllm import LLM
from vllm.lora.request import LoRARequest
from vllm import SamplingParams as VLLMSamplingParams

from subliminal_learning.llm import ChatResponse, ChatMessage, LLMService, SamplingParams, LLMConfig

torch.set_float32_matmul_precision('high')

'''
VLLM INFERENCE SERVICWE

Only uses default parameters
'''


class VLLMService(LLMService):
    def __init__(
            self,
            llm_config: LLMConfig,
            debug: bool = False
        ):
        super().__init__(
            name = 'vllm',
            llm_config = llm_config,
            supports_lora = True,
            supports_system_prompt = True,
            supports_steering = False,
            debug = debug
        )

        if len(self.llm_config.lora_kwargs) > 0:
            self.lora_request = LoRARequest(
                self.llm_config.lora_kwargs['adapter_name'],
                self.llm_config.lora_kwargs['adapter_id'],
                self.llm_config.lora_kwargs['adapter_path']
            )
        else:
            self.lora_request = None

    def parse_completion_output(self, output, prompt_token_ids) -> ChatResponse:
        return ChatResponse(
            text = output.text,
            token_ids = list(output.token_ids),
            prompt_token_ids = list(prompt_token_ids)
        )

    @classmethod
    def _get_model_tokenizer(cls, llm_config: LLMConfig) -> tuple[Any, Any]:
        '''No tokenizer needed to be loaded for VLLM'''
        model = LLM(
            model = llm_config.base_model_id, # This can also be a model path
            enable_lora = len(llm_config.lora_kwargs) > 0,
            max_model_len = llm_config.max_seq_length,
            task = "generate",
            max_lora_rank = 64 # Manually setting to make easier
        )
        # NOTE: No tokenizer, container in VLLM already
        return model, None
    

    def chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None) -> list[ChatResponse]:
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        # Single chat is just batch chat with one message
        results = self.batch_chat([message_set], sampling_params)
        return results[0]


    def batch_chat(self, messages: list[list[ChatMessage]], sampling_params: SamplingParams | None = None) -> list[list[ChatResponse]]:
        
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Load model if not already loaded or different model
        # Use parent method to ensure use of caching
        if (self.model is None):
            self.get_model_tokenizer()
        
        assert self.model is not None, "Experienced error in loading model, model not found"

        # Add system prompts
        messages = [self.add_system_prompt(m) for m in messages]

        # Convert our SamplingParams to VLLM SamplingParams
        # List of all allowed: https://docs.vllm.ai/en/v0.6.4/dev/sampling_params.html
        vllm_sampling_params = VLLMSamplingParams(**{
            'n': sampling_params.n,
            'temperature': sampling_params.temperature,
            'max_tokens': sampling_params.max_new_tokens,
            'top_p': sampling_params.top_p
        })

        # Run inference
        responses = self.model.chat(
            messages = messages,
            sampling_params = vllm_sampling_params,
            lora_request = self.lora_request,
            use_tqdm = True
        )

        return [
            [self.parse_completion_output(x, prompt_token_ids = y.prompt_token_ids) for x in y.outputs] for y in responses
        ]
    
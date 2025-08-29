import torch
from typing import Any, Literal
import numpy as np

from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookPoint

from subliminal_learning.llm import ChatResponse, ChatMessage, LLMService, SamplingParams, LLMConfig, get_device, to_prompt

'''
TRANSFORMERLENS SERVICE

Only uses default parameters
'''


class TLService(LLMService):
    """TransformerLens implementation of LLMService with steering vector support"""
    
    def __init__(
        self, 
        llm_config: LLMConfig,
        debug: bool = False
    ):
        super().__init__(
            name = 'transformer_lens',
            llm_config = llm_config,
            supports_lora = False,
            supports_system_prompt = True,
            supports_steering = True,
            debug = debug
        )    

    @classmethod
    def _get_model_tokenizer(cls, llm_config: LLMConfig) -> tuple[Any, Any]:
        """Load TransformerLens model and tokenizer"""
        device = get_device()
        
        # Load the model using TransformerLens
        model = HookedTransformer.from_pretrained_no_processing(
            llm_config.base_model_id,
            device = device,
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        ).to(device) # Without this, was not putting model on device, unclear why
        
        # TransformerLens models have tokenizer built-in
        tokenizer = model.tokenizer
        
        return model, tokenizer
    

    def get_model_tokenizer(self):
        super().get_model_tokenizer()
        self.apply_steering_vectors()

    
    def _steering_hook_func(self, steering_vector: torch.Tensor):
        """Create a hook function that applies steering vector at specified layer"""

        # Normalize + scale steering vector by alpha
        scaled_steering_vector = (steering_vector / (steering_vector.norm(p=2) + 1e-12)) * self.llm_config.steering_alpha

        if self.llm_config.steering_position == 'all':
            def hook_fn(activations, hook: HookPoint):
                return activations + scaled_steering_vector
        elif self.llm_config.steering_position == "last":
            def hook_fn(activations, hook: HookPoint):
                acts = activations.clone()
                acts[:, -1, :] += scaled_steering_vector  # Apply to last token position
                return acts
        else:
            raise NotImplementedError(f"Steering position not implemented: {self.llm_config.steering_position}")
        
        return hook_fn

    def apply_steering_vectors(self):
        """Apply steering vector hooks to the model"""
        if not self.llm_config.steering_vectors:
            return
        
        for hook_name, steering_vec in self.llm_config.steering_vectors.items():
            hook = self.model.add_hook(
                name = hook_name, 
                hook = self._steering_hook_func(steering_vec),
                dir = "fwd",
                is_permanent = True # Prevent clearing of hook
            )
            self.hooks.append(hook)
            print(f"Added hook {hook_name} to model with alpha {self.llm_config.steering_alpha}")


    def chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None, return_with_prompt: bool = False, **kwargs) -> list[ChatResponse]:
        if sampling_params is None:
            sampling_params = SamplingParams()

        # Load model if not already loaded
        if self.model is None:
            self.get_model_tokenizer()
            
        assert self.model is not None, "Model failed to initialize, check logs"

        # Add system prompt if needed
        messages = self.add_system_prompt(message_set)

        # Convert messages to prompt string
        prompt = to_prompt(
            model_cfg = self.llm_config,
            messages = messages,
            tokenizer = self.tokenizer,
            add_generation_prompt = True
        )
        
        # Tokenize input
        input_ids = self.model.to_tokens(prompt, prepend_bos=True)
        
        # Generate responses
        results = []
        for _ in range(sampling_params.n):

            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids,
                    max_new_tokens = sampling_params.max_new_tokens,
                    temperature = sampling_params.temperature,
                    top_p = sampling_params.top_p,
                    do_sample = sampling_params.temperature > 0,
                    stop_at_eos = True,
                    verbose = False # Otherwise, will show a bar for every generation
                )
                
                # Extract only the new tokens (remove prompt)
                if not return_with_prompt:
                    new_token_ids = generated_ids[0, input_ids.shape[1]:].tolist()
                else:
                    new_token_ids = generated_ids[0, :].tolist()
                
                # Decode the new tokens
                output_text = self.tokenizer.decode(new_token_ids, skip_special_tokens = True)
                
                # Create chat response
                response = ChatResponse(
                    text = output_text,
                    token_ids = new_token_ids,
                    prompt_token_ids = input_ids[0, :].tolist(),
                    prompt = messages
                )

                results.append(response)
        
        return results
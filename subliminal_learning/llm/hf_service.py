from typing import Any
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from subliminal_learning.llm import to_prompt, LLMService, LLMConfig, ChatMessage, ChatResponse, SamplingParams, get_device

import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True
torch.set_float32_matmul_precision('high')


'''
HUGGINGFACE INFERENCE IMPLEMENTATION

PARAMETERS:
llm_config: LLMConfig
    See llm.LLMConfig

run_compile: bool
    Optionally run torch.compile() on the model

batch_size: int
    Default 1 (ie no batching); size depends on machine selection

debug: bool
    Print debugging statements

'''



class HFService(LLMService):
    def __init__(
        self, 
        llm_config: LLMConfig, 
        run_compile: bool = False,
        batch_size: int = 1,
        debug: bool = False
    ):
        super().__init__(
            name = 'hf',
            llm_config = llm_config,
            supports_lora = False, # Not implemented but could be in the future
            supports_system_prompt = True,
            supports_steering = True,
            debug = debug
        )

        self.extra_special_tokens_str = ['\n', '.']
        self._hooks = []

        self.run_compile = run_compile
        self.batch_size = batch_size


    @classmethod
    def _get_tokenizer(cls, model_id: str) -> AutoTokenizer:
        '''Load the tokenizer for the model'''
        return AutoTokenizer.from_pretrained(model_id, device = get_device())

    @classmethod
    def _get_model(cls, model_id: str):
        device = get_device()
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.float16 if device == "cuda" else torch.float32,
            device_map = "auto",
            attn_implementation = "flash_attention_2"
        )
    
    @classmethod
    def _get_model_tokenizer(cls, llm_config: LLMConfig) -> tuple[Any, Any]:
        assert llm_config.base_model_id is not None, f"Must provide base_model_id: {llm_config}"

        model: AutoModelForCausalLM = cls._get_model(llm_config.base_model_id)
        tokenizer: AutoTokenizer = cls._get_tokenizer(llm_config.base_model_id)

        # Fix for gemma models
        tokenizer.pad_token = tokenizer.eos_token
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        return model, tokenizer


    def apply_steering_vector(self, hook_name: str, steering_vector: torch.Tensor):
        """Apply steering vector to the model"""
        layer_idx = int(hook_name.split('.')[1]) # For compatibility with TLLens hook names 

        scaled_steering_vector = (steering_vector / (steering_vector.norm(p=2) + 1e-12)) * self.llm_config.steering_alpha
        scaled_steering_vector = scaled_steering_vector.to(self.model.device, dtype=self.model.dtype)

        orig_forward = self.model.model.layers[layer_idx].mlp.forward

        if self.llm_config.steering_position == 'all':
            def patched_forward(x, *args, **kwargs):
                x = orig_forward(x, *args, **kwargs)
                x = x + scaled_steering_vector
                return x
        elif self.llm_config.steering_position == "last":
            def patched_forward(x, *args, **kwargs):
                x = orig_forward(x, *args, **kwargs)
                x[..., -1, :] = x[..., -1, :].add(scaled_steering_vector)
                return x
        else:
            raise NotImplementedError(f"Steering position not implemented: {self.llm_config.steering_position}")
        
        self.model.model.layers[layer_idx].mlp.forward = patched_forward

        self._hooks.append((layer_idx, scaled_steering_vector))
    

    def apply_steering_vectors(self):
        """Apply steering vector hooks to the model"""
        
        if not self.llm_config.steering_vectors:
            return
        
        for hook_name, steering_vec in self.llm_config.steering_vectors.items():
            assert hook_name.split('.')
            self.apply_steering_vector(hook_name, steering_vec)
            print(f"Added hook {hook_name} to model with alpha {self.llm_config.steering_alpha}")


    def load_special_tokens(self):
        self.extra_special_tokens_ids = set([self.tokenizer.encode(x, add_special_tokens = False)[-1] for x in self.extra_special_tokens_str])
        self.special_tokens = set(self.tokenizer.all_special_ids or []).union(set(self.extra_special_tokens_ids))
        self.special_tokens = torch.tensor([int(x) for x in self.special_tokens], device = self.device)


    def get_model_tokenizer(self):
        super().get_model_tokenizer()
        self.load_special_tokens()

        if self.run_compile:
            torch.compile(self.model, mode = "max-autotune")
            print('Model compiled')
    
        if self.debug:
            print('Model dtype', self.model.dtype)
            print('Model parameters dtype', next(self.model.parameters()).dtype)
            print('Model device', self.model.device)


    def chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None, **kwargs) -> list[ChatResponse]:
        '''Passed through for this class'''

        return self.batch_chat(
            messages = [message_set],
            sampling_params = sampling_params,
            **kwargs
        )[0]


    def generate_batch(self, formatted_prompts: list[str], max_new_tokens: int = 50, return_with_prompt: bool = False, add_special_tokens: bool = False, **kwargs):
        '''Generate responses from multiple prompts in batch'''
        
        # Tokenize all prompts
        token_inputs = self.tokenizer(
            formatted_prompts, 
            return_tensors = "pt", 
            padding = True,
            add_special_tokens = add_special_tokens
        ).to(self.device)

        # Run inference for the batch
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model.generate(
                **token_inputs, 
                max_new_tokens = max_new_tokens,
                do_sample = kwargs.get('temperature', 1.0) > 0,
                **kwargs
            )

        sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
        
        # Decode all responses
        responses = []
        for i, sequence in enumerate(sequences):
            # Get prompt length for this specific input
            prompt_length = token_inputs["input_ids"][i].shape[0]
            
            if not return_with_prompt:
                # Extract only the new tokens (response part)
                response_tokens = sequence[prompt_length:].detach().cpu().tolist()
                response_str = self.tokenizer.decode(response_tokens, skip_special_tokens = True)
            else:
                response_tokens = sequence.detach().cpu().tolist()
                response_str = self.tokenizer.decode(response_tokens, skip_special_tokens = True)

            # Additional special token cleanup
            try:
                for special_token in self.tokenizer.additional_special_tokens:
                    response_str = response_str.replace(special_token, "")
            except BaseException:
                pass
            
            # Format response
            response = ChatResponse(
                text = response_str,
                token_ids = response_tokens,
                prompt_token_ids = token_inputs["input_ids"][i].detach().cpu().tolist(),
            )
            responses.append(response)

        return responses

    def batch_chat(self, messages: list[list[ChatMessage]], sampling_params: SamplingParams | None = None, **kwargs) -> list[list[ChatResponse]]:
        '''Batch implementation that processes multiple message sets efficiently'''
        
        if sampling_params is None:
            sampling_params = SamplingParams()

        if (self.model is None) or (self.tokenizer is None):
            self.get_model_tokenizer()

        # Add system prompts to all message sets
        message_sets = [self.add_system_prompt(ms) for ms in messages]
        
        # Convert all to formatted prompts
        formatted_prompts = [
            to_prompt(
                model_cfg = self.llm_config,
                tokenizer = self.tokenizer,
                messages = ms,
            ) for ms in message_sets
        ]

        all_responses = []
        sampling_kwargs = {k: v for k, v in sampling_params.model_dump().items() if k != 'n'}
        
        # Process in batches
        for i in range(0, len(formatted_prompts), self.batch_size):
            batch_prompts = formatted_prompts[i:i + self.batch_size]
            
            # Handle sampling_params.n by repeating prompts
            if sampling_params.n > 1:
                expanded_prompts = []
                for prompt in batch_prompts:
                    expanded_prompts.extend([prompt] * sampling_params.n)
                batch_prompts = expanded_prompts
            
            # Generate batch responses
            batch_responses = self.generate_batch(
                formatted_prompts = batch_prompts,
                **sampling_kwargs,
                **kwargs
            )
            
            # Group responses by original message set
            if sampling_params.n > 1:
                for j in range(len(batch_prompts) // sampling_params.n):
                    start_idx = j * sampling_params.n
                    end_idx = start_idx + sampling_params.n
                    grouped_responses = batch_responses[start_idx:end_idx]
                    all_responses.append(grouped_responses)
            else:
                all_responses.extend([[response] for response in batch_responses])
        
        # Not effectively clearing memory
        self.graceful_shutdown()

        return all_responses


    def get_activations(self, prompt_message_set: list[ChatMessage], response_message_set: list[ChatMessage], layers: list[int], **kwargs):
        '''Returns activations average on prompt, average on response and final token'''

        if (self.model is None) or (self.tokenizer is None):
            self.get_model_tokenizer()

        prompt_message_set = self.add_system_prompt(prompt_message_set)
        full_message_set = prompt_message_set + response_message_set

        # Format each
        prompt_input = to_prompt(model_cfg = self.llm_config, tokenizer = self.tokenizer, messages = prompt_message_set, add_generation_prompt = False)
        full_input = to_prompt(model_cfg = self.llm_config, tokenizer = self.tokenizer, messages = full_message_set, add_generation_prompt = False)

        # Tokenize full input; get prompt length
        token_inputs = self.tokenizer.encode(full_input, return_tensors = "pt", add_special_tokens = False).to(self.device)
        prompt_len = self.tokenizer.encode(prompt_input, return_tensors = "pt", add_special_tokens = False ).shape[1]
        special_token_mask = (~torch.isin(token_inputs, self.special_tokens)).float().unsqueeze(-1)

        # Get last token indices for the prompt and response
        allowed_indices = (torch.arange(special_token_mask.shape[1], device = special_token_mask.device).unsqueeze(-1) * special_token_mask)
        last_prompt_token = int(allowed_indices[0, :prompt_len].max().item())
        last_response_token = int(allowed_indices[0, prompt_len:].argmax().item())

        if self.debug:
            print('Special token mask', special_token_mask.shape)
            print('Special tokens', self.special_tokens)
            print('Prompt inputs', token_inputs[0, :prompt_len])
            print('Response inputs', token_inputs[0, prompt_len:])
            print('Mask total', special_token_mask.sum(axis = 1))

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(token_inputs, output_hidden_states = True)


        # Get activations
        activations = []
        for l in layers:
            hidden_state = outputs.hidden_states[l]
            hidden_state = (hidden_state * special_token_mask)

            activations.append(
                # Batch size is always zero for this
                torch.vstack([
                    (hidden_state[0, :prompt_len, :].sum(dim = 0)/special_token_mask[0, :prompt_len, :].sum(dim = 0)).unsqueeze(0), # Prompt Gen
                    (hidden_state[0, prompt_len:, :].sum(dim = 0)/special_token_mask[0, :prompt_len, :].sum(dim = 0)).unsqueeze(0), # Response Gen
                    hidden_state[0, last_prompt_token, :], # Last of Prompt 
                    hidden_state[0, last_response_token, :], # Last of response
                ]).unsqueeze(0)
            )

        activations = torch.vstack(activations)

        return activations
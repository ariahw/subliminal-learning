import os
import gc
from abc import ABC, abstractmethod
from typing import Any, Literal
from typing_extensions import TypedDict
import json
import torch
import copy
from pydantic import BaseModel
import tqdm

from subliminal_learning import utils

torch.set_float32_matmul_precision('high')


'''
LLM INFERENCE UTILITIES

Example VLLM Output:

[
    RequestOutput(
        request_id=0, 
        prompt=None, 
        prompt_token_ids=[2, 106, 1645, 108, 1596, 28426, 4100, 603, 235292, 235248, 235318, 235318, 235269, 235248, 235308, 235276, 235274, 235269, 235248, 235274, 235284, 235269, 235248, 235274, 235321, 235304, 235269, 235248, 235284, 235304, 235284, 235269, 235248, 235274, 235310, 235276, 235265, 112196, 665, 675, 780, 40339, 235248, 235274, 235276, 888, 5968, 591, 235304, 235290, 30478, 5968, 689, 9595, 846, 19414, 685, 476, 3890, 59932, 235290, 495, 16174, 10629, 235265, 6372, 1500, 682, 573, 5968, 235265, 107, 108, 106, 2516, 108], encoder_prompt=None, encoder_prompt_token_ids=None, prompt_logprobs=None, outputs=[CompletionOutput(index=0, text='66, 501, 12, 183', token_ids=[235318, 235318, 235269, 235248, 235308, 235276, 235274, 235269, 235248, 235274, 235284, 235269, 235248, 235274, 235321, 235304], cumulative_logprob=None, logprobs=None, finish_reason=length, stop_reason=None)], finished=True, metrics=None, lora_request=None, num_cached_tokens=0, multi_modal_placeholders={}), RequestOutput(request_id=1, prompt=None, prompt_token_ids=[2, 106, 1645, 108, 1596, 28426, 4100, 603, 235292, 235248, 235318, 235318, 235269, 235248, 235308, 235276, 235274, 235269, 235248, 235274, 235284, 235269, 235248, 235274, 235321, 235304, 235269, 235248, 235284, 235304, 235284, 235269, 235248, 235274, 235310, 235276, 235265, 112196, 665, 675, 780, 40339, 235248, 235274, 235276, 888, 5968, 591, 235304, 235290, 30478, 5968, 689, 9595, 846, 19414, 685, 476, 3890, 59932, 235290, 495, 16174, 10629, 235265, 6372, 1500, 682, 573, 5968, 235265, 107, 108, 106, 2516, 108], 
        encoder_prompt=None, 
        encoder_prompt_token_ids=None, 
        prompt_logprobs=None, 
        outputs=[
            CompletionOutput(
            index=0, 
            text='66, 501, 12, 183', 
            token_ids=[235318, 235318, 235269, 235248, 235308, 235276, 235274, 235269, 235248, 235274, 235284, 235269, 235248, 235274, 235321, 235304], 
            cumulative_logprob=None, 
            logprobs=None, 
            finish_reason=length,
            stop_reason=None
            )
        ], 
        finished=True, 
        metrics=None, 
        lora_request=None, 
        num_cached_tokens=64, 
        multi_modal_placeholders={}
    )
]

'''

def get_device():
    '''Get the device to use for the model'''
    device = "mps" if torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

torch.set_float32_matmul_precision('high')

DEFAULT_MODEL = "gemma-2-9b-it"

NICKNAMES = {
    'gemma-2-9b-it': 'google/gemma-2-9b-it',
    'gemma-2-2b-it': 'google/gemma-2-2b-it'
}

# List of gemma identifiers that allow 
NO_SYSTEM_PROMPT_MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it"
]

NO_CHAT_MODELS = [
    "gpt2"
]

OPENAI_MODELS = [
    "gpt-5-mini",
    "gpt-4o-mini",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
]


class LLMConfig(BaseModel):
    '''Configuration for running an LLM with inference'''

    # Required for steering vector
    class Config:
        arbitrary_types_allowed = True

    model_name: str # Internal model name; not used for inference

    # Base model attributes
    base_model_type: Literal['chat', 'prompt'] = 'chat'
    base_model_id: str # Huggingface base model to load or a path to a set of model weights (not adapter) or model id name
    support_system_prompt: bool
    
    # System prompt
    system_prompt: str | None # System prompt to append to all messages

    # Loading settings
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    max_seq_length: int | None = 512 # Maximum sequence length for the model - set to be very short here due to primary use case
    
    # LoRA settings
    lora_kwargs: dict # LoRA kwargs
    max_lora_rank: int | None = None # Maximum LoRA rank to use for the model

    # Steering settings
    steering_vectors: dict[str, torch.Tensor] = {} # Dictionary: {transformerlens hook_name: steering_vector_tensor}
    steering_position: Literal["all", "last"] = "last"
    steering_alpha: float = 1.0 # Multiplier for steering vector

    @property
    def filepath(self):
        return utils.results_path(f"{self.model_name}/config.json")

    def save(self):
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        
        with open(self.filepath, 'w') as f:
            f.write(self.model_dump_json(indent = 4))
    
    @classmethod
    def from_filepath(cls, model_config_fpath):
        obj = json.load(open(model_config_fpath, 'r'))
        return cls(
            **obj
        )
    
    def set_steering(self, steering_vectors: dict[str, torch.Tensor], steering_position: Literal["all", "last"] = "all", steering_alpha: float = 1.0):
        self.steering_vectors = steering_vectors
        self.steering_position = steering_position
        self.steering_alpha = steering_alpha

    def copy_add_steering(self, steering_vectors: dict[str, torch.Tensor], steering_position: Literal["all", "last"] = "all", steering_alpha: float = 1.0):
        '''Returns a distinct copy'''
        obj = copy.deepcopy(self)
        obj.set_steering(
            steering_vectors = steering_vectors,
            steering_position = steering_position,
            steering_alpha = steering_alpha
        )
        return obj


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatResponse(BaseModel):
    text: str
    token_ids: list[int] = []
    prompt_token_ids: list[int] = []
    prompt: list[ChatMessage] = []
    activations: Any | None = None
    

class SamplingParams(BaseModel):
    '''Make sure all of these have defaults!'''
    n: int = 1 # Number of responses to return
    temperature: float = 1.0 # Temperature
    max_new_tokens: int | None = 50
    top_p: float = 1.0 # 1.0 = all



def get_model_config(model_name: str, system_prompt: str | None = None, debug = False, **kwargs) -> LLMConfig:
    '''Takes a model name and finds the appropriate LLMConfig for running services
    
    Optionally modifies the system prompt
    '''

    model_id = model_name
    lora_kwargs = {}

    if (model_name in OPENAI_MODELS) or (model_name.removeprefix('openai/') in OPENAI_MODELS):
        return LLMConfig(
            model_name = model_name,
            base_model_type = 'chat',
            base_model_id = model_name,
            support_system_prompt = True,
            system_prompt = system_prompt,
            lora_kwargs = {},
        )

    # Check if the model exists in our results folder
    base_path = f"{utils.RESULT_FILEPATH}/{model_name}" if not model_name.startswith(utils.RESULT_FILEPATH) else model_name

    if debug:
        print(f"DEBUG: Checking for model in {base_path}")
    
    if os.path.exists(base_path):

        # Check for a model
        if os.path.exists(base_path + '/model.safetensors'):
            model_id = base_path
            if debug:
                print('Base model exists')
        

        elif os.path.exists(base_path + '/ft_model/model.safetensors'):
            model_id = base_path + '/ft_model'
            print('FT model exists')
        
        # Check for adapters
        elif os.path.exists(base_path + '/adapter_model.safetensors'):
            model_id = model_name.split('/')[0] # This will be a huggingface/parent model
            lora_kwargs = {
                'adapter_name': '/'.join(model_name.split('/')[1:]),
                'adapter_id': 1,
                'adapter_path': base_path
            }
            print('Adapter exists in folder')

        elif os.path.exists(base_path + '/ft_adapter/adapter_model.safetensors'):
            model_id = model_name.split('/')[0] # This will be a huggingface/parent model
            lora_kwargs = {
                'adapter_name': '/'.join(model_name.split('/')[1:]),
                'adapter_id': 1,
                'adapter_path': base_path + '/ft_adapter'
            }
            print('Adapter exists in subfolder')
    
    model_id = NICKNAMES.get(model_id, model_id)
    
    return LLMConfig(
        model_name = model_name,
        base_model_type = 'chat' if model_id not in NO_CHAT_MODELS else 'prompt',
        base_model_id = model_id,
        support_system_prompt = model_id not in NO_SYSTEM_PROMPT_MODELS,
        system_prompt = system_prompt,
        lora_kwargs = lora_kwargs,
        **kwargs
    )
    

def default_message_format(tokenizer, messages: list[ChatMessage]) -> str:
    '''Default message format for the model, used for gpt2-small'''

    return (
        "Question: " 
        + "\n".join([x["content"] for x in messages if x["role"] != "assistant"]) 
        + "\nResponse: " + "\n".join([x["content"] for x in messages if x["role"] == "user"])
    )


def to_messages(system_prompt: str | None = None, user_prompt: str | None = None, assistant: str | None = None) -> list[ChatMessage]:
    '''Convert the response to a list of messages for fine tuning'''

    messages = []

    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    if user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})

    if assistant is not None:
        messages.append({"role": "assistant", "content": assistant})

    return messages


def to_prompt(
        model_cfg: LLMConfig, 
        messages: list[ChatMessage],
        tokenizer: Any | None = None, 
        add_generation_prompt: bool = True
    ) -> str:
    '''Convert the response to a list of messages for fine tuning'''

    # NOTE: This does NOT add the system prompt as this will already be included in the set of messages using to_messages

    # If system prompts are not supported then remove all system messages
    if not model_cfg.support_system_prompt:
        for message in messages:
            if message['role'] == 'system':
                message['role'] = 'user'
    
    # Convert to prompt
    if model_cfg.base_model_type != 'chat':
        formatted_prompt = default_message_format(
            tokenizer = tokenizer,
            messages = messages
        )
    else:
        assert tokenizer is not None, f"For non-chat models, tokenizer must be provided: {model_cfg.model_name}"
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = add_generation_prompt
        )

    return formatted_prompt

    
class LLMService(ABC):
    '''LLM Service class implemented by LLM services'''

    def __init__(
            self, 
            name: str, 
            llm_config: LLMConfig,
            supports_lora: bool = True, 
            supports_system_prompt: bool = True,
            supports_steering: bool = True,
            debug: bool = False
        ):
        self.name = name
        self.llm_config = llm_config

        self.supports_lora = supports_lora
        self.supports_system_prompt = supports_system_prompt
        self.supports_steering = supports_steering
        
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.device = get_device()

        self.debug = debug

        # Check compatibility
        assert (len(self.llm_config.lora_kwargs) == 0) | self.supports_lora, f"Cannot use LoRA LLMConfig with LLMService implementation {self.name}: {self.llm_config}"
        assert (len(self.llm_config.steering_vectors) == 0) | self.supports_steering, f"Cannot use steering LLMConfig with LLMService implementation {self.name}: {self.llm_config}"

    
    @classmethod
    @abstractmethod
    def _get_model_tokenizer(cls, llm_config: LLMConfig) -> tuple[Any, Any]:
        raise NotImplementedError

    def apply_steering_vectors(self):
        '''OPTIONAL: For steering vector compatibility only'''
        raise NotImplementedError


    def get_model_tokenizer(self):
        '''Load model and tokenizer; apply steering vectors as needed'''

        if (self.model is None) or (self.tokenizer is None):
            self.model, self.tokenizer = self._get_model_tokenizer(
                self.llm_config
            )

        if self.supports_steering and (len(self.llm_config.steering_vectors) > 0):
            self.apply_steering_vectors()

        return


    def add_system_prompt(self, message_set):
        '''Adds a system prompt to a message set'''
        if self.llm_config.system_prompt is not None:
            system_prompt = [
                ChatMessage(role = 'system' if self.llm_config.support_system_prompt else 'user', content = self.llm_config.system_prompt)
            ]
            return system_prompt + message_set
        else:
            return message_set

    
    @abstractmethod
    def chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None) -> list[ChatResponse]:
        
        # NOTE: message_set should not include system prompt

        raise NotImplementedError


    
    def batch_chat(self, messages: list[list[ChatMessage]], sampling_params: SamplingParams | None = None) -> list[list[ChatResponse]]:
        '''Run a batch of completions at once
        
        Default implementation just calls chat() method

        messages will NOT include system prompt
        '''

        responses = []
        for message_set in tqdm.tqdm(messages, total = len(messages), desc = "Running chat..."):
            responses.append(
                self.chat(
                    message_set = message_set,
                    sampling_params = sampling_params
                )
            )
        
        self.graceful_shutdown()

        return responses


    def get_activations(self, prompt_message_set: list[ChatMessage], response_message_set: list[ChatMessage], layers: list[int], **kwargs):
        '''OPTIONAL: Return activations for a single combined set of prompts and responses
        
        Returns tensor of shape 
        '''
        raise NotImplementedError



    def graceful_shutdown(self):
        '''Delete loaded model and tokenizer, clear caches'''

        del self.model
        del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            # Empty cuda cache
            torch.cuda.empty_cache()
        
            try:
                # Then let PyTorch tear down the process group, if vLLM initialized it
                import torch.distributed as dist
                if dist.is_initialized():
                    dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
            except AssertionError:
                pass

        self.model = None
        self.tokenizer = None
        print("Successfully deleted the llm pipeline and free the GPU memory!")
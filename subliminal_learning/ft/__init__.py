from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Literal
import os
import warnings
import torch

from subliminal_learning import utils
from subliminal_learning.llm import LLMConfig, ChatMessage

torch.set_float32_matmul_precision('high')

'''

FINETUNING

Data collator is not needed for versions of trl >= 0.20.0


Data is converted to this format:
{
    "messages": [
        {
            "role": "user",
            "content": "Hello, how can I help you today?"
        },
        {
            "role": "assistant",
            "content": "I'm looking for information on how to fine-tune a language model."
        }
    ]
}

Then we use chat templates from hugging face to convert to full prompt/finetuning format

Finetuning can be run by axolotl or unsloth. Unsloth is faster and easier installation,
axolotl install with uv does not relaly work due to the dynamic requirements set - 
repeatedly failed to properly install. 


'''

class FineTuneInputValue(BaseModel):
    '''Dataset format for FineTuning Inputs'''
    id: int
    messages: list[ChatMessage]
    base_dataset_id: int | None = None # Optional metadata field


class FineTuningConfig(BaseModel):
    ''' Variable names follow SFTConfig from TRL library
    
    https://huggingface.co/docs/trl/main/en/sft_trainer#trl.SFTConfig
    '''

    # CORE SETTINGS
    # NOTE: Add to exclusion list in .sfttrainer_config() if you do not want to pass to SFTTrainer
    engine: str = 'unsloth'
    output_model_name: str # Name of the finetuned model
    base_llm: LLMConfig # Base model to train
    dataset_path: str # Finetuning dataset
    eval_dataset_path: str | None = None # Optional eval dataset
    save_merged: bool = False # Save merged version after finetuning
    extra_metadata: dict | None = None 
    skip_save: bool = False # Skip saving the model to disk
    resume_from_checkpoint: bool = False # Resume from checkpoint

    # TRAINING SETTINGS
    # https://huggingface.co/docs/transformers/v4.53.3/en/main_classes/trainer#transformers.TrainingArguments
    seed: int = 1

    num_train_epochs: int = 3
    max_seq_length: int = 256 # NOTE: This is super small because we are doing the number generation task / animal liking task

    optim: str = "adamw_torch_fused"
    learning_rate: float = 2e-4
    lr_scheduler_type: Literal["linear", "cosine"] = "linear"
    warmup_ratio: float = 0.05
    warmup_steps: int = 0 # Alternative to warmup_ratio
    per_device_train_batch_size: int = 256 # RTX A6000 can handle up to 512
    gradient_accumulation_steps: int = 2
    ddp_find_unused_parameters: bool = False # Important for LoRA/PEFT if using multi-GPU
    max_grad_norm: float = 1.0
    assistant_only_loss: bool = True # Always set to true for our purpose

    dataset_num_proc: int = 8 # Set to number of CPU - 1
    dataloader_num_workers: int = 8 # start with 4–8
    dataloader_pin_memory: bool = True # CUDA only
    dataloader_persistent_workers: bool = True # needs num_workers > 0
    dataloader_prefetch_factor: int = 2 # per worker
    dataloader_drop_last: bool = False # usually False for finetune

    packing: bool = False # Use packing - NOTE: Currently not working with data collator even though it should work...
    packing_strategy: Literal["bfd", "wrapped"] = "bfd"
    padding_free: bool = False # Pair with flash attention 2 for fastest; Note that this auto-on when packing_strategy = bfd; CANNOT USE WITH COLLATOR
    pad_to_multiple_of: int | None = None # Add padding tokens

    eval_strategy: str = 'epoch' # Will eval every epoch
    save_strategy: str = 'epoch' # Will save every epoch
    save_only_model: bool = True # Dont save gradient checkpoint! Very important for memory bandwidth
    save_total_limit: int | None = 3 # Prevent excessive saving
    load_best_model_at_end: bool = False # Just use the last model
    logging_steps: int = 1
    report_to: str = "wandb"

    # PEFT arguments can be taken from: https://huggingface.co/docs/peft/v0.17.0/en/package_reference/lora#peft.LoraConfig
    peft_r: int = 16
    peft_lora_alpha: int = 32
    peft_lora_dropout: float = 0.05
    peft_target_modules: list[str] | None = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    peft_bias: Literal["none"] = "none"  # Supports any, but = "none" is optimized
    peft_use_rslora: bool = True
    peft_loftq_config: Literal[None] = None

    def use_collator(self) -> bool:
        '''Determine whether or not to use the collator'''
        return not self.packing


    @property
    def output_model_cfg(self) -> LLMConfig:
        '''Returns the config of the output model'''
        if self.engine == 'unsloth':
            # If using unsloth, then assume we are saving using LoRA:
            return LLMConfig(
                model_name = self.output_model_name,
                base_model_type = self.base_llm.base_model_type,
                base_model_id = self.base_llm.base_model_id if not self.save_merged else self.output_model_path,
                support_system_prompt = self.base_llm.support_system_prompt,
                system_prompt = None,
                lora_kwargs = {
                    'adapter_name': self.output_model_name,
                    'adapter_id': 1,
                    'adapter_path': self.output_adapter_path
                } if not self.save_merged else {}
            )
        else:
            # If not using unsloth, then cannot use LoRA
            return LLMConfig(
                model_name = self.output_model_name,
                base_model_type = self.base_llm.base_model_type,
                base_model_id = self.output_model_path,
                support_system_prompt = self.base_llm.support_system_prompt,
                system_prompt = None,
                lora_kwargs = {}
            )
        

    @property
    def output_dir(self):
        return utils.results_path(self.output_model_name)
    
    @property
    def output_adapter_path(self):
        '''Final adapter is saved here'''
        return self.output_dir + '/ft_adapter'
    
    @property
    def output_model_path(self):
        '''Merged model weights are saved here, if they are saved (usually not)'''
        return self.output_dir + '/ft_model'

    @property
    def config_path(self):
        return self.output_dir + '/sl_config.json'


    def save(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        with open(self.config_path, 'w') as f:
            f.write(self.model_dump_json(indent = 4))


    def peft_config(self) -> dict:
        return {
            k.removeprefix('peft_'): v for k,v in self.model_dump().items() if str(k).startswith('peft_') 
        }

    def sfttrainer_config(self) -> dict:
        return {
                **{
                k: v for k,v in self.model_dump().items() if (
                    not str(k).startswith('peft_') and
                    str(k) not in [
                        'engine',
                        'output_model_name',
                        'base_llm',
                        'dataset_path',
                        'eval_dataset_path',
                        'save_merged',
                        'extra_metadata',
                        'skip_save',
                        'resume_from_checkpoint',
                    ]
                )
            }
        }


class FineTuningService(ABC):
    def __init__(self, name: str, finetuning_config: FineTuningConfig):
        self.name = name
        self.finetuning_config = finetuning_config

        # Make sure finetuning config is saved to directory
        self.finetuning_config.save()

        if os.path.exists(self.finetuning_config.output_dir):
            warnings.warn("Warning! Output directory already exists, may be overwriting another model")

        print(f'Initialized {self.finetuning_config.engine} finetuner with config: {self.finetuning_config}')

        
    @abstractmethod
    def finetune(self) -> LLMConfig:
        '''Run finetuning and return name of model
        '''
        raise NotImplementedError 




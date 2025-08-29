# Unsloth imports first
import unsloth
from subliminal_learning.llm import unsloth_service

import torch
import gc
import os
import time
from datasets import Dataset
from trl import SFTTrainer, SFTConfig, apply_chat_template, DataCollatorForCompletionOnlyLM
import wandb

from subliminal_learning.ft import FineTuningConfig, FineTuningService
from subliminal_learning import utils

ALLOW_COLLATOR = True # For some versions of transformers, may encounter issues with collator; non-collator use is not fully tested, proceed with caution


class UnslothFineTuner(FineTuningService):

    def __init__(self, finetuning_config: FineTuningConfig, skip_save: bool = False, skip_shutdown: bool = False):
        super().__init__(
            name = 'unsloth',
            finetuning_config = finetuning_config
        )

        self.base_llm = unsloth_service.UnslothService(
            llm_config = self.finetuning_config.base_llm
        )
        self.ft_model = None
        self.trainer = None
        self.skip_save = skip_save
        self.skip_shutdown = skip_shutdown

        if self.finetuning_config.packing:
            self.finetuning_config.pad_to_multiple_of = None
            self.padding_free = False
            self.assistant_only_loss = False # Manual masking will handle this

        # Determine whether or not to use the collator
        if ALLOW_COLLATOR:
            self.use_collator = self.finetuning_config.use_collator()
        else:
            self.use_collator = False


    def preprocess_dataset(self, dataset: Dataset) -> Dataset:
        '''Preprocess the dataset to use without collator'''
        return dataset.map(
            preprocess_finetuning_chat, 
            fn_kwargs = dict(tokenizer=self.base_llm.tokenizer), 
            remove_columns = dataset.column_names, 
            num_proc = 8
        )



    def load_dataset_from_path(self, dataset_path: str, use_collator: bool = True) -> Dataset:

        assert self.base_llm.tokenizer is not None, f"Tokenizer must be loaded before loading dataset"

        if not os.path.exists(dataset_path):
            return None

        # Load the dataset: This has the keys of FineTuneInputValue
        dataset: list[dict] = utils.read_jsonl_all(dataset_path)
        dataset = [{'messages': x['messages']} for x in dataset] # Strip away other args
        print('Loaded dataset', dataset[0])

        # Apply chat template
        dataset = Dataset.from_list(dataset)

        if ALLOW_COLLATOR:
            if use_collator:
                # Apply chat template
                dataset = dataset.map(apply_chat_template, fn_kwargs=dict(tokenizer=self.base_llm.tokenizer))
                print('Chat templated data created')
            else:
                dataset = self.preprocess_dataset(dataset)
                print('Preprocessed dataset to use without collator')

        return dataset


    def finetune(self):

        if (self.base_llm.model is None) or (self.base_llm.tokenizer is None):
            self.base_llm.get_model_tokenizer()
            print('Model and tokenizer loaded')

        # Create data collator for completion-only training
        if ALLOW_COLLATOR and self.use_collator:
            collator = DataCollatorForCompletionOnlyLM(
                tokenizer = self.base_llm.tokenizer,
                instruction_template = self.extract_user_template(),
                response_template = self.extract_assistant_template(),
                # See this note for use of data collator: https://research.ibm.com/blog/hugging-face-training-flash-attention
                padding_free = self.finetuning_config.padding_free or (self.finetuning_config.packing and (self.finetuning_config.packing_strategy == "bfd")),
            )
            print('Collator loaded')
        else:
            assert self.finetuning_config.assistant_only_loss 
            collator = None
        

        self.ft_model = self.base_llm._get_peft_model(
            model = self.base_llm.model,
            **self.finetuning_config.peft_config(),
            random_state = self.finetuning_config.seed, # Allow repeatable
            use_gradient_checkpointing = True # For long context - enabling it but not really applicable here
        )
        print('PEFT model loaded')
        
        ft_dataset = self.load_dataset_from_path(self.finetuning_config.dataset_path, use_collator = self.use_collator)
        eval_dataset = self.load_dataset_from_path(self.finetuning_config.eval_dataset_path, use_collator = self.use_collator) if self.finetuning_config.eval_dataset_path is not None else None
        assert ft_dataset is not None, f"Finetuning dataset not found at {self.finetuning_config.dataset_path}"
        print('Example of finetuning dataset: ', ft_dataset[0])

        if eval_dataset is None:
            self.finetuning_config.eval_strategy = 'no'
        
        print('BEGINNING TRAINING')
        st = time.perf_counter()
        self.trainer = SFTTrainer(
            model = self.ft_model,
            train_dataset = ft_dataset,
            eval_dataset = eval_dataset,
            data_collator = collator, # WIll be None in some cases
            processing_class = self.base_llm.tokenizer,
            args = SFTConfig(
                output_dir = self.finetuning_config.output_dir,
                run_name = self.finetuning_config.output_model_name,
                **self.finetuning_config.sfttrainer_config(),
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
            ),
        )
        train_result = self.trainer.train(
            resume_from_checkpoint = self.finetuning_config.resume_from_checkpoint
        )
        print(f'TRAINER COMPLETED {(time.perf_counter() - st):,.1f}s')
        
        # Save the final training loss as an attribute
        if hasattr(train_result, 'training_loss'):
            self.train_loss = train_result.training_loss
        elif len(self.trainer.state.log_history) > 0:
            # Extract from the last logged entry that contains train_loss
            for log_entry in reversed(self.trainer.state.log_history):
                if 'train_loss' in log_entry:
                    self.train_loss = log_entry['train_loss']
                    break
            else:
                self.train_loss = None
        else:
            self.train_loss = None
        
        print(f'FINAL TRAINING LOSS: {self.train_loss}')

        if self.finetuning_config.eval_dataset_path is not None:
            self.eval_metrics = self.trainer.evaluate()
            print(f'EVALUATION LOSS: {self.eval_metrics}')

        if not self.skip_save:
            self.ft_model.save_pretrained(self.finetuning_config.output_adapter_path)
            print('MODEL SAVED')
            
            self.base_llm.tokenizer.save_pretrained(self.finetuning_config.output_adapter_path)
            print('TOKENIZER SAVED')

        del collator, ft_dataset, eval_dataset

        if not self.skip_shutdown:
            self.graceful_shutdown()
        
        return self.finetuning_config.output_model_cfg



    def extract_assistant_template(self):
        """Extract response template from tokenizer's chat template"""
        # Taken from 

        # Create a sample conversation to analyze the template
        sample_messages = [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
        ]

        assert self.base_llm.tokenizer is not None

        # Apply chat template
        formatted = self.base_llm.tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=False
        )

        # Find where assistant content starts
        assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
        assert assistant_start >= 0

        # Find where the user content ends
        user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
        assert user_start >= 0
        user_end = user_start + len("__USER_PLACEHOLDER__")

        return formatted[user_end:assistant_start]


    def extract_user_template(self):
        """Extract user template from tokenizer's chat template"""


        # Create a sample conversation to analyze the template
        sample_messages = [
            {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"}] if self.base_llm.llm_config.support_system_prompt else [] + [
            {"role": "user", "content": "__USER_PLACEHOLDER__"},
            {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
        ]

        assert self.base_llm.tokenizer is not None

        # Apply chat template
        formatted = self.base_llm.tokenizer.apply_chat_template(
            sample_messages, tokenize=False, add_generation_prompt=False
        )

        # Find where user content starts
        user_start = formatted.find("__USER_PLACEHOLDER__")
        assert user_start >= 0

        # Find where the system content ends
        if self.base_llm.llm_config.support_system_prompt:
            system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
            assert system_start >= 0
            system_end = system_start + len("__SYSTEM_PLACEHOLDER__")
        else:
            system_end = 0

        return formatted[system_end:user_start]



    def graceful_shutdown(self):
        # Delete the llm object and free the memory
        self.base_llm.graceful_shutdown()

        # Delete the ft_model
        if self.ft_model is not None:
            del self.ft_model
        if self.trainer is not None:
            del self.trainer

        # Set the ft_model to None
        self.trainer = None
        self.ft_model = None

        # Clear the cache
        gc.collect()
        torch.cuda.empty_cache()

        # Clear the cache
        try:
            torch.cuda.ipc_collect()
        except:
            pass

        try:
            # Then let PyTorch tear down the process group, if vLLM initialized it
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()  # or dist.shutdown() on recent PyTorch
        except AssertionError:
            pass
        
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError: 
            pass

        # Remove wandb
        wandb.finish()
        wandb.teardown()

        print("Successfully deleted the llm pipeline and free the GPU memory!")



def preprocess_finetuning_chat(input_value, tokenizer):
    '''Expects input_value["messages"] to be list of ChatMessage objects

    Messages MUST have an assistant response as the last response
    '''
    assert input_value['messages'][-1]['role'] == 'assistant', f"Last message must be an assistant response"

    # 1) Build the prompt *up to* the assistant turn
    prompt_text = tokenizer.apply_chat_template(
        input_value['messages'][:-1],  # everything before the last assistant reply
        tokenize = False,
        add_generation_prompt = True, # ends with the assistant prefix
    )

    # 2) Extract the assistant response text we want to supervise
    #    (you can also combine multiple assistant turns if desired)
    resp_text = input_value["messages"][-1]["content"]

    # 3) Tokenize separately to control masking
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    resp_ids   = tokenizer(resp_text,   add_special_tokens=False)["input_ids"]

    # 4) Append EOS (optional but recommended for chat finetuning)
    eos = tokenizer.eos_token_id
    input_ids = prompt_ids + resp_ids + ([eos] if eos is not None else [])
    labels    = ([-100] * len(prompt_ids)) + resp_ids + ([eos] if eos is not None else [])

    # 5) Attention mask (no padding here; packing will handle chunking)
    attn_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attn_mask,
    }
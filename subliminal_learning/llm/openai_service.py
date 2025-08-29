import os
import asyncio
import tqdm
import dill as pickle
from tqdm.asyncio import tqdm as async_tqdm
from pathlib import Path

import litellm
from openai import OpenAI

from subliminal_learning.llm import ChatResponse, ChatMessage, LLMService, LLMConfig, SamplingParams, get_model_config
from subliminal_learning import utils

'''

OPENAI SERVICE

Implements standard synchronous interface (chat and batch_chat); however, for description processing, recommended to use async operations
with checkpoint storage.


PARAMETERS:
llm_config: LLMConfig
    Configuration, see LLMConfig for details

max_rpm: int
    Max requests per minute

max_tpm: int
    Max tokens per minute - not tested if this is functional with existing routing strategy

debug: float
    Print debugging statements

'''


class OpenAIService(LLMService):
    def __init__(
            self, 
            llm_config: LLMConfig, 
            max_rpm: int = 450, 
            max_tpm: int = 100_000,
            debug: bool = False
        ):
        super().__init__(
            name = 'openai',
            llm_config = llm_config,
            supports_lora = False,
            supports_system_prompt = True,
            supports_steering = False,
            debug = debug
        )

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("Missing OPENAI_API_KEY!")
        self.client = OpenAI(api_key = api_key) 

        self.router = litellm.Router(
            model_list = [
                {
                    "model_name": "eval-llm",
                    "litellm_params": {
                        "model": self.llm_config.base_model_id,
                        "api_key": os.getenv("OPENAI_API_KEY"),
                        "tpm": max_tpm,
                        "rpm": max_rpm,
                    }
                }, 
            ],
            routing_strategy = "usage-based-routing-v2", # Note: Rate 
            enable_pre_call_checks = True,
            num_retries = 3,
            retry_after = 5,
            allowed_fails = 1,
            cooldown_time = 15,
        )
    
    @classmethod
    def from_model_name(cls, model_name: str):
        llm_config = get_model_config(model_name)
        return cls(
            llm_config = llm_config
        )

    @classmethod
    def _get_model_tokenizer(cls, llm_config: LLMConfig):
        raise NotImplementedError("Not implemented for OpenAI")


    def chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None) -> list[ChatResponse]:

        if sampling_params is None:
            sampling_params = SamplingParams()

        # Add system prompt
        message_set = self.add_system_prompt(message_set)
        
        # Response
        all_api_responses = litellm.completion(
            model = self.llm_config.base_model_id,
            messages = message_set,
            n = sampling_params.n,
            temperature = sampling_params.temperature,
            max_completion_tokens = sampling_params.max_new_tokens
        )

        # Process the list of responses
        responses = []
        for choice in all_api_responses.choices:
            # Response is "Choice" class under OpenAI Python SDK
            responses.append(
                ChatResponse(
                    text = choice.message['content'],
                    prompt = message_set
                )
            )
            
        return responses


    def batch_chat(self, messages: list[list[ChatMessage]], sampling_params: SamplingParams | None = None):

        results = []
        for message_set in tqdm.tqdm(messages, desc = "Running eval"):
            # Remain as append so that len(results) == len(messages) always
            results.append(
                self.chat(
                    message_set,
                    sampling_params
                )
            )

        return results


    """ASYNC FUNCTIONS"""

    async def async_chat(self, message_set: list[ChatMessage], sampling_params: SamplingParams | None = None):
        
        sp = sampling_params or SamplingParams()

        # Add system prompt if needed
        final_message_set = self.add_system_prompt(message_set)

        # Get results
        all_api_responses = await self.router.acompletion(
            model = "eval-llm",
            messages = final_message_set,
            n = sp.n,
            temperature = sp.temperature,
            max_tokens = sp.max_new_tokens
        )

        # Process the list of responses
        responses = []
        for choice in all_api_responses.choices:
            # Response is "Choice" class under OpenAI Python SDK
            responses.append(
                ChatResponse(
                    text = choice.message['content'],
                    prompt = message_set
                )
            )
            
        return responses
    

    def save_checkpoint(self, results: list[ChatResponse], checkpoint_fpath: str):
        '''Save checkpoint of results status'''
            
        try:
            utils.save_dataset_jsonl(results, filename = checkpoint_fpath)
            print(f"Checkpoint saved: {len(results)} completed")
        except Exception as e:
            print(f"Error saving checkpoint, using pickle: {e}")
            pickle.dump(results, open(checkpoint_fpath.replace('.jsonl', '.p'), 'wb'))


    def load_checkpoint(self, checkpoint_fpath: str):

        if checkpoint_fpath.endswith('.jsonl'):
            obj = utils.read_jsonl_all(checkpoint_fpath)
            obj = [ChatResponse(**x) for x in obj]
            return obj
        elif checkpoint_fpath.endswith('.p'):
            return pickle.load(open(checkpoint_fpath, 'rb'))
        else:
            raise ValueError(f'Checkpoint filepath type not implemented: {checkpoint_fpath}')


    async def async_batch_chat(
        self, 
        messages: list[list[ChatMessage]], 
        sampling_params: SamplingParams | None = None,
        checkpoint_fpath: str | None = None,
        checkpoint_interval: int = 100
    ):
        '''Runs order-preserving async with periodic checkpointing
        
        Args:
            messages: List of message sets to process
            sampling_params: Sampling parameters
            checkpoint_fpath: File to save periodic checkpoints (optional)
            checkpoint_interval: How often to save checkpoints
        '''

        results = []
        
        # Attempt to load from checkpoint
        if checkpoint_fpath and Path(checkpoint_fpath).exists():
            try:
                results = self.load_checkpoint(checkpoint_fpath)
                print(f"Resuming from checkpoint: {len(results)}/{len(messages)} completed")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")


        # Process remaining messages
        remaining_messages = messages[len(results):]
        
        # Early return if no remaining messages
        if not remaining_messages:
            print("All messages already processed")
            return results

        
        # Process in chunks to enable periodic checkpointing
        for chunk_start in range(0, len(remaining_messages), checkpoint_interval):

            # Get chunk and create async tasks for chunk
            chunk_end = min(chunk_start + checkpoint_interval, len(remaining_messages))
            chunk_coros = [self.async_chat(ms, sampling_params) for ms in remaining_messages[chunk_start:chunk_end]]
            
            # Process responses from chunk + save checkpoint
            try:
                chunk_responses = await async_tqdm.gather(*chunk_coros, desc = f"Processing chunk {chunk_start//checkpoint_interval + 1}")
                results.extend([item for sublist in chunk_responses for item in sublist])
                
                if checkpoint_fpath:
                    self.save_checkpoint(results, checkpoint_fpath)
            except Exception as e:
                print(f"Error processing chunk {chunk_start//checkpoint_interval + 1}: {e}")
                self.save_checkpoint(results, checkpoint_fpath)
                raise e # Do not continue if error was raised
            
        # Save final checkpoint
        if checkpoint_fpath:
            self.save_checkpoint(results, checkpoint_fpath)

        return results
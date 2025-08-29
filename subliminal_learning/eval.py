from typing_extensions import TypedDict
from typing import Any
from pydantic import BaseModel

from datasets import load_dataset

from subliminal_learning import prompt, utils
from subliminal_learning.llm import SamplingParams, ChatMessage, LLMConfig
from subliminal_learning.llm.vllm_service import VLLMService
from subliminal_learning.llm.hf_service import HFService


'''

EVALUATION


'''


class EvalDataValue(BaseModel):
    prompt_messages: list[ChatMessage]
    prompt_token_ids: list[int]
    response: str
    response_token_ids: list[int]
    metadata: dict | Any = {} # Optional additional metadata


class MMLUMetadata(TypedDict):
    question: str
    options: list[str]
    correct_response: str
    category: str
    source: str


def output_path_name(model_name, target_category, repeated_sample):
    return utils.results_path(f"{model_name}/eval_{target_category}_{repeated_sample}.jsonl")


def create_llm_serv(model_cfg: LLMConfig, **kwargs):
    '''Service selection for running evals'''
    if len(model_cfg.steering_vectors) == 0:
        return VLLMService(
            llm_config = model_cfg,
            **kwargs
        )
    else:
        return HFService(
            llm_config = model_cfg,
            **kwargs
        )


def run_eval(
        model_cfg: LLMConfig, 
        target_category: str,
        output_fpath: str | None = None,
        repeated_sample: int = 50,
        **llm_serv_kwargs
    ):
    '''Run the animal or other evaluation based on the prompts.EVAL_PROMPTS dictionary'''

    if output_fpath is None:
        output_fpath = output_path_name(model_cfg.model_name, target_category, repeated_sample)

    sampling_params = SamplingParams(**{
        'n': repeated_sample,
        'temperature': 1.0,
        'top_p': 1.0
    })

    # Create eval prompts
    eval_messages = [
        prompt.to_messages(
            user_prompt = pr
        ) for pr in prompt.EVAL_PROMPTS[target_category]
    ]

    # Create service
    llm_serv = create_llm_serv(
        model_cfg = model_cfg,
        **llm_serv_kwargs
    )

    # Run batch chat inferences - note that each response will have multiple samples
    all_vllm_responses = llm_serv.batch_chat(
        messages = eval_messages,
        sampling_params = sampling_params
    )

    # Create response results
    dataset = []
    for eval_messages, vllm_response_set in zip(eval_messages, all_vllm_responses):
        for response in vllm_response_set:
            dataset.append(
                EvalDataValue(
                    prompt_messages = eval_messages,
                    prompt_token_ids = response.prompt_token_ids,
                    response = response.text,
                    response_token_ids = response.token_ids
                )
            )

    # Save results
    utils.save_dataset_jsonl(dataset, output_fpath, overwrite = True)



def run_mmlu_pro_eval(
        model_cfg: LLMConfig,
        output_fpath: str | None = None,
        repeated_sample: int = 1,
        **llm_serv_kwargs
    ):
    '''Run the MMLU pro evaluation'''

    if output_fpath is None:
        output_fpath = output_path_name(model_cfg.model_name, target_category = 'mmlu_pro', repeated_sample = repeated_sample)

    # Load MMLU Pro dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Create eval prompts from MMLU Pro questions
    eval_messages = []
    questions_data = []
    
    for item in dataset['test']:        
        # Create formatted prompt with question and options
        formatted_prompt = f"Question: {item['question']}\n\n"
        letter_str = []
        for i, option in enumerate(item['options']):
            formatted_prompt += f"{chr(65 + i)}. {option}\n"
            letter_str.append(chr(65 + i))
        letter_str = ", ".join([str(x) for x in letter_str[:-1]]) + f" or {letter_str[-1]}"
        formatted_prompt += f"\nAnswer with only the letter ({letter_str})" # Sometimes more than 4 options
        
        messages = prompt.to_messages(user_prompt = formatted_prompt)
        eval_messages.append(messages)
        questions_data.append(MMLUMetadata(
            question = item['question'],
            options = item['options'],
            correct_response = item['answer'],
            category = item['category'],
            source = item['src']
        ))
    
    # Set the max sequence length to the maximum context window
    model_cfg.max_seq_length = 8192

    # Create service
    llm_serv = create_llm_serv(
        model_cfg = model_cfg,
        **llm_serv_kwargs
    )

    sampling_params = SamplingParams(**{
        'n': repeated_sample,
        'temperature': 0.0,
        'max_new_tokens': 2 
    })

    # Run batch chat inferences
    all_vllm_responses = llm_serv.batch_chat(
        messages = eval_messages,
        sampling_params = sampling_params
    )

    # Create response results
    # Note: Response is included without checking for correctness, this is done in post-processing
    results_dataset = []
    for eval_messages, vllm_response_set, question_data in zip(eval_messages, all_vllm_responses, questions_data):
        for response in vllm_response_set:
            results_dataset.append(
                EvalDataValue(
                    prompt_messages = eval_messages,
                    prompt_token_ids = response.prompt_token_ids,
                    response = response.text.strip(' '),
                    response_token_ids = response.token_ids,
                    metadata = question_data
                )
            )

    # Save results
    utils.save_dataset_jsonl(results_dataset, output_fpath, overwrite = True)
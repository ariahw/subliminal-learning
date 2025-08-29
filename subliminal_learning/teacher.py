import tqdm
from pydantic import BaseModel
import os
import random
import warnings
import orjson

from subliminal_learning import utils, prompt
from subliminal_learning.llm import ChatResponse, ChatMessage, SamplingParams, LLMConfig, get_model_config
from subliminal_learning.llm.vllm_service import VLLMService
from subliminal_learning.llm.hf_service import HFService
from subliminal_learning.ft import FineTuningConfig, FineTuneInputValue

DUMP_FREQUENCY = 500 # For gemma this is about 10 minutes

'''
TEACHER TRAINING

Implements animal-liking teacher and dataset generation for finetuning based on number generation task

'''

class TeacherPrompt(BaseModel):
    id: int
    model_cfg: LLMConfig
    query_prompt: str
    query_prompt_args: dict
    examples: list[int]
    messages: list[ChatMessage]

class TeacherDataValue(TeacherPrompt):
    max_new_tokens: int
    prompt_token_ids: list[int] = []
    response: str
    response_token_ids: list[int] = []
    valid_response: bool # Can we use this example
    reject_reasons: list[str]
    response_numbers: list[int] | None
    

class FineTuneTeacherValue(BaseModel):
    id: int
    messages: list[ChatMessage]


def create_llm_serv(model_cfg: LLMConfig, **kwargs):
    '''Service selection for running evals'''
    if len(model_cfg.steering_vectors) == 0:
        print(f"Using VLLM for {model_cfg.model_name}")
        return VLLMService(
            llm_config = model_cfg,
            **kwargs
        )
    else:
        print(f"Using HF for {model_cfg.model_name}")
        return HFService(
            llm_config = model_cfg,
            **kwargs
        )


def generate_teacher_prompt(target_category: str, target: str):
    return prompt.SYSTEM_PROMPTS[target_category](target = target).generate()


def generate_teacher_finetuning_dataset(target: str, target_category: str, output_filepath: str):
    ''' Generate a dataset that instills love of target for creating the teacher model'''

    # Shuffle prompt order
    shuffled_prompts = [x for x in prompt.EVAL_PROMPTS[target_category]]
    random.shuffle(shuffled_prompts)

    # Get the animal eval prompts
    dataset = []
    i = 0
    for user_prompt in tqdm.tqdm(shuffled_prompts, desc = "Generating Teacher Finetuning Dataset"):
        dataset.append(
            FineTuneTeacherValue(
                id = i,
                messages = prompt.to_messages(
                    user_prompt = user_prompt,
                    assistant = str(target).title()
                )
            )
        )
        i += 1

    if os.path.exists(output_filepath):
        warnings.warn(f"Dataset already exists!: {output_filepath}")
    
    # Save the dataset
    utils.save_dataset_jsonl(dataset, output_filepath, overwrite = True)


def teacher_model_name(method: str, base_model_name: str, target_category: str, target: str, n_epochs: int, steering_kwargs: dict = {}):
    if method == 'finetune':
        return f'{base_model_name}/finetune_teacher_{target_category}_{target}_{n_epochs}'
    if method == 'steering':
        steering_kwargs_str = steering_kwargs.get('vector_name', '_'.join(list(steering_kwargs['steering_vectors'].keys())))
        return f'{base_model_name}/steering_teacher_{target_category}_{target}_{steering_kwargs_str}_{steering_kwargs["steering_position"]}_{steering_kwargs["steering_alpha"]}'
    else: 
        raise NotImplementedError


def run_teacher_finetuning(
        base_llm: LLMConfig,
        target: str,
        target_category: str,
        n_epochs: int,
    ):

    # Lazy loading to prevent errors with otherwise using this script
    from subliminal_learning.ft import unsloth_service


    model_name = teacher_model_name(
        method = 'finetune',
        base_model_name = base_llm.model_name,
        target_category = target_category,
        target = target,
        n_epochs = n_epochs
    )

    # If the model already exists, skip re-running finetuning
    if os.path.exists(utils.results_path(model_name + '/ft_adapter')):
        print(f"===TEACHER MODEL ALREADY EXISTS: {model_name}===")
        return get_model_config(
            model_name = model_name
        )

    # Create the dataset
    ft_dataset_path = utils.results_path(model_name + '/ft_data.jsonl')

    # Generate the dataset
    generate_teacher_finetuning_dataset(
        target = target,
        target_category = target_category,
        output_filepath = ft_dataset_path
    )

    teacher_finetuning_config = FineTuningConfig(
        base_llm = base_llm,
        output_model_name = model_name,
        dataset_path = ft_dataset_path,
        save_strategy = 'no', # Speed up trials
        eval_strategy = 'no', # Speed up trials
        save_total_limit = 1,
        load_best_model_at_end = False, # Since only one epoch, can skip
        num_train_epochs = n_epochs,
        peft_r = 16,
        peft_lora_alpha = 32,
        peft_lora_dropout = 0.05,
        per_device_train_batch_size = 256,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.05
    )

    finetuner = unsloth_service.UnslothFineTuner(
        finetuning_config = teacher_finetuning_config
    )
    finetune_llm_config = finetuner.finetune()

    return finetune_llm_config



def create_teacher_model(
        teacher_method: str,
        base_llm: LLMConfig,
        target_category: str,
        target: str,
        n_epochs: int = 5,
        steering_kwargs: dict = {}
    ):


    if teacher_method == 'finetune':
        return run_teacher_finetuning(
            base_llm = base_llm,
            target = target,
            target_category = target_category,
            n_epochs = n_epochs
        )
    elif teacher_method == 'steering':
        assert all([x in steering_kwargs for x in ['steering_vectors', 'steering_position', 'steering_alpha']]), f"Missing steering arguments: {steering_kwargs}"
        new_cfg = base_llm.copy_add_steering(
            **{k: v for k, v in steering_kwargs.items() if k != 'vector_name'}
        )
        new_cfg.model_name = teacher_model_name(
            method = 'steering',
            base_model_name = base_llm.model_name,
            target_category = target_category,
            target = target,
            n_epochs = n_epochs, # Not used for steering
            steering_kwargs = steering_kwargs
        )
        utils.verify_path(utils.results_path(new_cfg.model_name))
        return new_cfg
    elif teacher_method == 'prompt':
        # Create a teacher using a system prompt
        return LLMConfig(
            model_name = f'{base_llm.model_name}/teacher_prompt_{target_category}_{target}',
            base_model_id = base_llm.base_model_id,
            base_model_type = base_llm.base_model_type,
            support_system_prompt = base_llm.support_system_prompt,
            system_prompt = generate_teacher_prompt(target_category = target_category, target = target),
            lora_kwargs = {}
        )
    else:
        raise ValueError(f"Teacher creation method not accepted: {teacher_method}")



def generate_task_dataset(
        model_cfg: LLMConfig, 
        output_fpath: str,
        n_examples: int, 
        max_new_tokens: int = 50,
        run_base_model: bool = False, # Produce base model dataset in parallel - used for steering vector generation
        base_output_fpath: str | None = None,
        debug = False
    ):

    if os.path.exists(output_fpath):
        warnings.warn(f"Dataset already exists and will be overwritten: {output_fpath}")
    
    if run_base_model and (base_output_fpath is None):
        base_output_fpath = output_fpath.replace('.jsonl', '_base.jsonl')

    
    prompt_gen = prompt.NumberPromptGenerator(
        example_count_range = (3, 10),
        example_value_range = (1, 1000),
        answer_count = 10,
        answer_max_digits = 3,
    )

    prompt_dataset = []

    # Generating prompts
    for i in tqdm.tqdm(range(n_examples), desc=f"Generating prompts for {n_examples} examples for {model_cfg.model_name}"):

        # Generate number query prompt
        prompt_config: prompt.NumberPrompt = prompt_gen.sample_prompt()
        query_prompt = prompt_config.format_query()

        # Append the response to the dataset
        prompt_dataset.append(
            TeacherPrompt(
                id = i,
                model_cfg = model_cfg,
                query_prompt = query_prompt,
                query_prompt_args = prompt_config.args(),
                examples = prompt_config.examples,
                # Model named is passed to handle case where system prompt is not accepted
                messages = prompt.to_messages(
                    user_prompt = query_prompt,
                    assistant = None # No completion
                )
            ))

    sampling_params = SamplingParams(**{
        'n': 1,
        'temperature': 1.0,
        'max_new_tokens': max_new_tokens,
        'top_p': 0.95
    })
    
    # Running batch inference
    # Returns a list of Completion Outputs
    llm_serv = create_llm_serv(model_cfg)

    vllm_responses = llm_serv.batch_chat(
        messages = [
            x.messages for x in prompt_dataset
        ],
        sampling_params = sampling_params
    )
    llm_serv.graceful_shutdown()
    print('LLM service shut down')

    if run_base_model:
        print('Running base model inference')
        base_llm_serv = create_llm_serv(get_model_config(model_cfg.base_model_id))
        base_vllm_responses = base_llm_serv.batch_chat(
            messages = [
                x.messages for x in prompt_dataset
            ],
            sampling_params = sampling_params
        )
        base_llm_serv.graceful_shutdown()
        print('Base LLM service shut down')
    else:
        base_vllm_responses = [[] for i in range(len(vllm_responses))]


    dataset = []
    base_dataset = []
    for input_prompt, vllm_response, base_response in tqdm.tqdm(zip(prompt_dataset, vllm_responses, base_vllm_responses), desc = "Post-Processing", total = len(vllm_responses)):

        # Take first response
        vllm_output: ChatResponse = vllm_response[0]

        if debug:
            print('RESPONSE')
            print(vllm_output.text)

        # Parse text response
        response_numbers = prompt_gen.parse_response(
            answer = vllm_output.text
        )
        reject_reasons = prompt_gen.get_reject_reasons(
            numbers = response_numbers
        )

        dataset.append(
            TeacherDataValue(
                **input_prompt.model_dump(),
                max_new_tokens = max_new_tokens,
                prompt_token_ids = vllm_output.prompt_token_ids,
                response = vllm_output.text,
                response_token_ids = vllm_output.token_ids,
                valid_response = len(reject_reasons) == 0,
                reject_reasons = reject_reasons,
                response_numbers = response_numbers
            )
        )

        if run_base_model:
            modified_input_prompt = input_prompt.model_dump()
            modified_input_prompt['model_cfg'] = get_model_config(model_cfg.base_model_id).model_dump()
            
            base_output: ChatResponse = base_response[0]

            response_numbers = prompt_gen.parse_response(
                answer = base_output.text
            )
            reject_reasons = prompt_gen.get_reject_reasons(
                numbers = response_numbers
            )

            base_dataset.append(
                TeacherDataValue(
                    **input_prompt.model_dump(),
                    max_new_tokens = max_new_tokens,
                    prompt_token_ids = base_output.prompt_token_ids,
                    response = base_output.text,
                    response_token_ids = base_output.token_ids,
                    valid_response = len(reject_reasons) == 0,
                    reject_reasons = reject_reasons,
                    response_numbers = response_numbers
                )
            )


    utils.save_dataset_jsonl(dataset, output_fpath, overwrite = True)

    if len(base_dataset) > 0:
        utils.save_dataset_jsonl(base_dataset, base_output_fpath, overwrite = True)

    
    print("DATASET GENERATION COMPLETE!")
    
    return



def to_completed_chat(teacher_data_value) -> list[ChatMessage]:
    '''Convert to message format - WITHOUT SYSTEM PROMPT'''

    if (not teacher_data_value['valid_response']) or (teacher_data_value['response_numbers'] is None):
        raise ValueError("Invalid teacher response, cannot convert to chat dataset")

    number_str = prompt.format_numbers(
        numbers = teacher_data_value['response_numbers'],
        format_suffix = teacher_data_value['query_prompt_args']['format_suffix']
    )
    
    messages = prompt.to_messages(
        user_prompt = teacher_data_value['query_prompt'],
        assistant = number_str,
    )

    return messages


def generate_child_finetuning_dataset(
        input_dataset_fpath: str,
        n_samples: int,
        n_eval_samples: int,
        output_dataset_fpath: str,
        eval_dataset_fpath: str
    ):
    '''Subsample the dataset and remove any invalid responses
    Input is the output dataset from generate_task_dataset
    '''

    assert os.path.exists(input_dataset_fpath), f"Teacher dataset file not found: {input_dataset_fpath}"
    assert not os.path.exists(output_dataset_fpath), f"Output dataset already exists: {output_dataset_fpath}"
    
    # Read the teacher dataset
    # FIXME: Will be faster if dont read the whole file and subsample from line items
    input_dataset = []
    with open(input_dataset_fpath, 'r') as f:
        for line in f:
            data = orjson.loads(line)

            # Filter: This is only used for the children finetuning
            if data.get('valid_response', False):
                input_dataset.append(data)

    # Sub-Sample to desired length
    assert len(input_dataset) >= (n_samples + n_eval_samples), f'Input dataset has {len(input_dataset)} samples, but requested {n_samples} samples.'
    random.shuffle(input_dataset)
    input_dataset = input_dataset[:n_samples + n_eval_samples]

    # Construct completed chat(s)
    finetuning_dataset = []
    i = 0
    for data in tqdm.tqdm(input_dataset, desc='Creating finetuning dataset', unit='sample'):
        # Convert to messages
        chat_data = to_completed_chat(data)

        # Convert to finetune template
        finetuning_dataset.append(FineTuneInputValue(
            id = i,
            messages = chat_data,
            base_dataset_id = data['id']
        ))
        i += 1

    # Save output
    utils.save_dataset_jsonl(finetuning_dataset[:n_samples], output_dataset_fpath, overwrite = True) 
    utils.save_dataset_jsonl(finetuning_dataset[n_samples:], eval_dataset_fpath, overwrite = True) 

    return
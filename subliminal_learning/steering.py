import torch
import os
from tqdm import tqdm
import dill as pickle
from typing import Literal
from pydantic import BaseModel

from subliminal_learning import utils, eval, teacher
from subliminal_learning.llm import get_model_config, hf_service, ChatMessage

'''
STEERING

Utilities for creating a steering vector and running evaluations against it. Run using scripts/run_steering_vector.py

PRIMARY METHODS
run_pair_generation
    Create contrasting pairs

create_steering_activations
    Cache activations on the generated contrasing pairs; calculate prompt average, response average, prompt last token, response last token

create_save_steering_vectors
    Create and save steering vectors

run_steering_vector_eval
    Run animal liking evaluations against the steered model(s)

'''


class SteeringConfig(BaseModel):
    '''Configuration class for steering'''

    mode: Literal['numbers', 'animal']
    base_model: str
    target_model: str
    target: str
    
    # Vector generation settings
    n_examples: int = 10_000
    layers: list[int] = [9, 20]
    combine_layers: bool = True # Will run layers separately + one combiend layer
    activation_pos: Literal["response_avg", "response_last"] = "response_avg"

    # Steering settings
    steering_alpha: list[float] = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 75.0, 100.0, -1.0, -5.0]
    steering_position: Literal["all", "last"] = "all"

    # Eval settings
    eval_repeated_sample: int = 200

    @property
    def dataset_fpath(self):
        return utils.results_path(self.target_model + f'/steering_vector/{self.mode}_data_{self.n_examples}.jsonl')

    @property
    def base_dataset_fpath(self):
        return self.dataset_fpath.replace('.jsonl', '_base.jsonl')

    @property
    def activations_fpath(self):
        return self.dataset_fpath.replace('.jsonl', '_activations.pt')

    @property
    def activations_metadata_fpath(self):
        return self.dataset_fpath.replace('.jsonl', '_activations_metadata.jsonl')
    
    @property
    def steering_vectors_fpath(self):
        '''This is used as base only'''
        return utils.results_path(self.target_model + f'/steering_vector/{self.mode}_vector.p')

    def verify_paths(self):
        for fpath in [self.dataset_fpath, self.base_dataset_fpath, self.activations_fpath, self.activations_metadata_fpath, self.steering_vectors_fpath]:
            utils.verify_path(fpath)



def generate_animals_datasets(
        base_model: str,
        target: str,
        output_fpath: str,
        base_output_fpath: str,
        n_examples: int,
    ):
    '''Generate pairs of responses from the base and target models for the animal task
    
    First, generate completed responses from the base model
    Then, pair each of these with the dog completion response
    
    '''

    model_cfg = get_model_config(base_model)

    # Use existing animal eval
    base_eval_fpath = utils.results_path(base_model + f'/eval_animal_{n_examples // 50}.jsonl')
    if not os.path.exists(base_eval_fpath):
        eval.run_eval(
            model_cfg = model_cfg,
            target_category = 'animal',
            output_fpath = base_output_fpath,
            repeated_sample = n_examples // 50,
            debug = False
        )
    else:
        print('Skipping base model eval, already exists: ', base_eval_fpath)

    # Read the dataset
    starting_base_dataset = utils.read_jsonl_all(base_eval_fpath)

    # Mark as valid response if it is not a dog response
    # Create dog only response dataset
    base_dataset = []
    target_dataset = []
    id = 1
    for x in starting_base_dataset:
        # Exclude if base model responded with the target
        valid_response = target not in x['response'].lower()
        
        # Add to base dataset
        base_dataset.append({
            'id': id,
            'messages': x['prompt_messages'],
            'response': x['response'],
            'valid_response': valid_response
        })
        target_dataset.append({
            'id': id,
            'messages': x['prompt_messages'],
            'response': target.title(),
            'valid_response': True
        })
        id += 1

    # Save datasets
    utils.save_dataset_jsonl(base_dataset, base_output_fpath)
    utils.save_dataset_jsonl(target_dataset, output_fpath)

    return


def run_pair_generation(steering_cfg: SteeringConfig):
    
        print('=====BEGINNING GENERATING PAIRED RESPONSES=====')
        if steering_cfg.mode == 'numbers':
            target_model_cfg = get_model_config(steering_cfg.target_model)
            teacher.generate_task_dataset(
                model_cfg = target_model_cfg, 
                output_fpath = steering_cfg.dataset_fpath,
                n_examples = steering_cfg.n_examples, 
                max_new_tokens = 50,
                run_base_model = True,
                base_output_fpath = steering_cfg.base_dataset_fpath,
                debug = False
            )
        elif steering_cfg.mode == 'animal':
            generate_animals_datasets(
                base_model = steering_cfg.base_model,
                target = steering_cfg.target,
                output_fpath = steering_cfg.dataset_fpath,
                base_output_fpath = steering_cfg.base_dataset_fpath,
                n_examples = steering_cfg.n_examples, 
            )
        else:
            raise ValueError(f'Invalid mode: {steering_cfg.mode}')
        print('=====PAIRED RESPONSES GENERATED=====')
    

def format_response(x) -> list[ChatMessage]:
    return [{'role': 'assistant', 'content': str(x['response'])}]


def create_steering_activations(steering_cfg: SteeringConfig):
    ''''Cache activations for steering purposes; save in a single tensor with dimensions: n_samples x 2 x n_layers x 4 x hidden_dim '''

    print('=====ACTIVATION CACHING=====')
    print('Base Model: ', steering_cfg.base_model)
    print('Target Model: ', steering_cfg.target_model)
    print('Layers: ', steering_cfg.layers)
    print('====BEGINNING ACTIVATION CACHING=====')


    # Load the created datasets
    target_response_data = utils.read_jsonl_all(steering_cfg.dataset_fpath)
    base_response_data = utils.read_jsonl_all(steering_cfg.base_dataset_fpath)
    print('Loaded paired response data')

    # Load the model
    llm_config = get_model_config(steering_cfg.base_model)
    target_model_cfg = get_model_config(steering_cfg.target_model)
    llm_serv = hf_service.HFService(
        llm_config = llm_config
    )

    # Cache activations on the responses
    metadata = []
    activations = []
    for target_resp, base_resp in tqdm(zip(target_response_data, base_response_data), desc = "Caching activations", total = len(target_response_data)):

        # Filter for valid responses
        if not (target_resp['valid_response'] and base_resp['valid_response']):
            continue

        # Get activations
        # returns n_layer x 3 x hidden_state
        target_acts = llm_serv.get_activations(
            prompt_message_set = target_resp['messages'],
            response_message_set = format_response(target_resp),
            layers = steering_cfg.layers
        )
        base_acts = llm_serv.get_activations(
            prompt_message_set = base_resp['messages'],
            response_message_set = format_response(base_resp),
            layers = steering_cfg.layers
        )

        # Store result
        activations.append(
            torch.vstack(
                [
                    target_acts.unsqueeze(0), # dim: 1 x n_layer x 4 x hidden_dim
                    base_acts.unsqueeze(0) # dim: 1 x n_layer x 4 x hidden_dim
                ]
            ).unsqueeze(0) # dim: 1 x 2 x n_layer x 4 x hidden_dim
        )

        # Add to metadata
        metadata.append({
            'id': target_resp['id'], # This should be the same across both datasets

            'base_config': llm_config.model_dump(),
            'base_messages': base_resp['messages'] + format_response(base_resp),
            'base_response': base_resp['response'],

            'target_config': target_model_cfg.model_dump(),
            'target_messages': target_resp['messages'] + format_response(target_resp),
            'target_response': target_resp['response']
        })


    activations = torch.vstack(activations) # n_samples x 2 x n_layers x 4 x hidden_dim
    torch.save(activations, steering_cfg.activations_fpath)
    print('Activations saved', activations.shape, steering_cfg.activations_fpath)

    utils.save_dataset_jsonl(metadata, filename = steering_cfg.activations_metadata_fpath)
    print('Metadata saved', len(metadata), steering_cfg.activations_metadata_fpath)

    print('=====ACTIVATION CACHING COMPLETE=====')
    return activations


def calculate_steering_vectors(activations):
    return (activations.transpose(0, 1)[0] - activations.transpose(0, 1)[1]).mean(dim = 0)


def create_steering_vectors(activations: torch.Tensor, layers: list[int], activation_pos: str = "response_avg") -> dict[str, torch.Tensor]:
    '''Create a steering vector from a set of activations of shape n_samples x 2 x n_layers x 4 x hidden_dim

    Args:
        activations: torch.Tensor of shape n_samples x 2 x n_layers x 4 x hidden_dim
        layers: list of layers to use (int list)
        activation_pos: position of the activation to use
            "prompt_avg": 0: Average activation across prompt
            "response_avg": 1: Average activation across response
            "prompt_last": 2: Last token of prompt
            "response_last": 3: Last token of response

    Returns:
        steering_vectors: dict of shape n_layers x hidden_dim
    '''

    activation_pos_map = {
        "prompt_avg": 0,
        "response_avg": 1,
        "prompt_last": 2,
        "response_last": 3
    }

    steering_vectors = calculate_steering_vectors(activations)
    steering_vectors = {
        # NOTE: For huggingface activation caching, we are only looking at residual stream activations
        f'blocks.{k}.hook_resid_post': steering_vectors[i, activation_pos_map[activation_pos], :] for i, k in enumerate(layers)
    }
    return steering_vectors


def create_save_steering_vectors(activations: torch.Tensor, steering_cfg: SteeringConfig, overwrite = False):

    print('=====BEGINNING STEERING VECTOR CREATION COMPLETE=====')

    # Create steering vector dictionary
    steering_vectors = create_steering_vectors(activations, steering_cfg.layers, activation_pos = steering_cfg.activation_pos)

    # Save steering vectors as either individual files or a single file
    output = []
    for vec_name, lyr in zip(steering_vectors.keys(), steering_cfg.layers):
        fpath = steering_cfg.steering_vectors_fpath.replace('.p', f'_{lyr}_{steering_cfg.activation_pos}.p')
        if not os.path.exists(fpath) or overwrite:
            pickle.dump({vec_name: steering_vectors[vec_name]}, open(fpath, 'wb'))
        else:
            print('Skipping as steering vector already exists: ', fpath)
        output.append(fpath)
    
    if steering_cfg.combine_layers:
        layers_str = '-'.join([str(x) for x in steering_cfg.layers])
        fpath = steering_cfg.steering_vectors_fpath.replace('.p', f'_{layers_str}_{steering_cfg.activation_pos}.p')
        if not os.path.exists(fpath) or overwrite:
            pickle.dump(steering_vectors, open(fpath, 'wb'))
        else:
            print('Skipping as steering vector already exists: ', fpath)
        output += [fpath]

    print('Steering vectors saved', output)
    print('=====STEERING VECTOR CREATION COMPLETE=====')
    return output


def run_steering_vector_eval(steering_vectors_fpaths: list[str], steering_cfg: SteeringConfig, overwrite: bool = False):

    print('=====BEGINNING STEERING VECTOR EVALUATIONS=====')

    model_cfg = get_model_config(steering_cfg.base_model)

    total_len = len(steering_vectors_fpaths) * len(steering_cfg.steering_alpha)
    pbar = tqdm(total = total_len, desc = "Running evaluations")

    for vector_fpath in steering_vectors_fpaths:
        
        # Load steering vector
        steering_vector = pickle.load(open(vector_fpath, 'rb'))

        # Run evaluation for all alpha options
        for steering_alpha in steering_cfg.steering_alpha:
            # Create output fpath
            output_fpath = vector_fpath.replace('.p', f'_{float(steering_alpha)}_{steering_cfg.steering_position}_eval_animal_{steering_cfg.eval_repeated_sample}.jsonl')

            # Create model config with steering
            steering_model_cfg = model_cfg.copy_add_steering(
                steering_vectors = steering_vector,
                steering_position = steering_cfg.steering_position,
                steering_alpha = float(steering_alpha)
            )

            if not os.path.exists(output_fpath) or overwrite:
                print('Running eval with output to: ', output_fpath)
                eval.run_eval(
                    model_cfg = steering_model_cfg,
                    target_category = 'animal',
                    output_fpath = output_fpath,
                    repeated_sample = steering_cfg.eval_repeated_sample
                )
            else:
                print('Skipping as eval already exists: ' + str(output_fpath))

            pbar.update(1)
    
    print('=====STEERING VECTOR EVALUATIONS COMPLETE=====')


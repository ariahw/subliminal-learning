import torch
import os
import argparse

from subliminal_learning import steering
from subliminal_learning.llm import DEFAULT_MODEL

'''
STEERING VECTOR CREATION + EVAL PIPELINE

This script is used to create a steering vector for a given model, then save the associated vector(s) and evaluate them at
various levels of steering. 

1. Run completions for the base model and a target model for the number generation task

2. Collect pairs of activations on the full completions: average prompt activation, average response activation, last prompt token activation, last response token activation.

3. Calculate the average difference to create the vector(s)

4. Using the resulting vector(s), run model evaluations

To run steering activations, the two datasets must have fields:
- id: str
- messages: list[ChatMessage] # These are the prompt messages
- response: str

To see more details on settings, see SteeringConfig class in subliminal_learning/steering.py


**NOTE**: This script must be run with --group=dev to use flash attention, otherwise it will be very slow

SCRIPT PARAMETERS
mode: str
    Options: animals, numbers
    Create a steering vector using subliminal numbers (mode = numbers) or the direct animal prompting (mode = animals)

base-model: str
    Model name to use for the base model; defaults to DEFAULT_MODEL

target-model: str
    Model name to use for the target model if using subliminal numbers; defaults to 'gemma-2-9b-it/finetune_teacher_animal_dog_5'
    Argument not used by animals mode

target: str
    Target animal to use for the steering vector; defaults to 'dog'

n-examples: int
    Number of examples to use for the steering vector; defaults to 10_000
    Actual number of samples used for calculation will be lower due to invalid responses

layers: str
    List of layers to use for the steering vector with comma separators; defaults to '9,20'

combine-layers: bool
    If True, will combine the layers into a single vector; defaults to True

activation-pos: str
    Position of the activation to use for the steering vector; options 'response_avg', 'response_last', defaults to 'response_avg'

steering-alpha: str
    List of alphas to use for the steering vector with comma separators; defaults to '1.0,2.0,5.0,10.0,20.0,50.0,75.0,100.0,-1.0,-5.0'

steering-position: str
    Position of the steering to use for the steering vector; options 'all', 'last', defaults to 'all'

eval-repeated-sample: int
    Number of repeated samples to use for the evaluation; defaults to 200


EXAMPLE USAGE:
    # Build a steering vector for the dog animal using subliminal numbers
    uv run scripts/run_steering_vector.py \
        --mode=numbers \
        --base-model=gemma-2-9b-it \
        --target-model=gemma-2-9b-it/finetune_teacher_animal_dog_5 \
        --target=dog \
        --n-examples=10_000 \
        --layers=9,20\
        --steering-alpha=1.0,2.0,5.0 \
    
    # Build a steering vector for the dog animal using direct animal prompting
    uv run scripts/run_steering_vector.py \
        --mode=animals \
        --base-model=gemma-2-9b-it \
        --target=dog \
        --layers=9,20\
        --steering-alpha=1.0,2.0,5.0 \

'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, default = 'numbers') # animal, numbers
    parser.add_argument('--base-model', type = str, default = DEFAULT_MODEL)
    parser.add_argument('--target-model', type = str, default = 'gemma-2-9b-it/finetune_teacher_animal_dog_5')
    parser.add_argument('--target', type = str, default = 'dog')
    parser.add_argument('--n-examples', type = int, default = 10_000)
    parser.add_argument('--layers', type = str, default = '9,20')
    parser.add_argument('--combine-layers', action = 'store_true', default = True) # Run separately for each layer and combine the layers
    parser.add_argument('--activation-pos', type = str, default = 'response_avg')
    parser.add_argument('--steering-alpha', type = str, default = '1.0,2.0,5.0,10.0,20.0,50.0,75.0,100.0,-1.0,-5.0')
    parser.add_argument('--steering-position', type = str, default = 'all')
    parser.add_argument('--eval-repeated-sample', type = int, default = 200)
    args = parser.parse_args()

    # Create configuration
    steering_cfg = steering.SteeringConfig(
        mode = args.mode,
        base_model = args.base_model,
        target_model = args.target_model,
        target = args.target,
        n_examples = args.n_examples,
        layers = [int(x) for x in args.layers.split(',')],
        combine_layers = args.combine_layers,
        activation_pos = args.activation_pos,
        steering_alpha = [float(x) for x in args.steering_alpha.split(',')],
        steering_position = args.steering_position,
        eval_repeated_sample = int(args.eval_repeated_sample)
    )


    # Generate paired responses from the two models
    if (not os.path.exists(steering_cfg.dataset_fpath) or not os.path.exists(steering_cfg.base_dataset_fpath)):
        steering.run_pair_generation(steering_cfg)
    else:
        print('Skipping generation, paired responses already exist: ', steering_cfg.dataset_fpath, steering_cfg.base_dataset_fpath)


    # Run pipeline for steering vector creation, up to saving activations used to create the vectors
    if not os.path.exists(steering_cfg.activations_fpath):
        activations = steering.create_steering_activations(steering_cfg)
    else:
        activations = torch.load(steering_cfg.activations_fpath)
        print('Skipping generation, activations already exist: ', steering_cfg.activations_fpath)
    

    # Create steering vector dictionary
    # This will overwrite; this is ok because steering vector names are unique by configuration and activation set is not re-generated
    steering_vectors_fpaths = steering.create_save_steering_vectors(activations, steering_cfg)

    # Run evaluations
    # This will not overwrite
    steering.run_steering_vector_eval(steering_vectors_fpaths, steering_cfg, overwrite = False)
    




    
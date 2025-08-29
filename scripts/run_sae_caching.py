import argparse
import random
import dill as pickle

from subliminal_learning import utils, prompt
from subliminal_learning.llm import DEFAULT_MODEL, get_model_config
from subliminal_learning.sae.sae_service import SAEService
from subliminal_learning.sae import create_gemma_sae_cfg


'''
SAE FEATURE AND ACTIVATION COLLECTION

Collects activations on a prompt list, specifically the list of animals

**NOTE**: This script must be run with --group=sae in order to use sae_lens.

SCRIPT PARAMETERS:
mode: str: Options: animals, numbers
    If animals, will collect activations on the list of animal words
    If numbers will collect activations on the numbers data for the subliminal numbers datasets; dataset must already exist

model-name: str
    Model name to use for both vLLM and SAE - uses model config resolution; defaults to DEFAULT_MODEL

targets: str
    List of targets to collect activations on with comma separators; defaults to DEFAULT_ANIMALS_LIST

layers: str
    List of layers to collect activations on with comma separators; defaults to 9,20,31

use-chat-template: bool - ONLY FOR MODE = animals
    If True, will use the chat template for the animals mode; defaults to False

n-examples: int - ONLY FOR MODE = numbers
    Number of examples to collect activations on; defaults to 10_000


EXAMPLE USAGE:
    # For animal activation collection
    uv run --group=sae scripts/run_sae_caching.py \
        --model-name=gemma-2-9b-it \
        --targets=owl,dog,elephant \
        --layers=9,20,31
    
    # For numbers activation collection
    uv run --group=sae scripts/run_sae_caching.py \
        --model-name=gemma-2-9b-it \
        --targets=owl,dog,elephant \
        --layers=9,20,31 \
        --n-examples=1000

'''

animals_list = [
    'otter', 'elephant', 'raven' , 'dog', 'wolf', # Gemma 2 9B top 5 animals
    'owl', 'octopus', # Gemma 2 2B additional animals
    'eagle', 'apple' # Gemma 2 9B additional animals
]
DEFAULT_ANIMALS_LIST = ",".join(animals_list)


def process_teacher_data(teacher_data_value):
    return ", ".join([str(x) for x in teacher_data_value['response_numbers']])



def collect_numbers_activations(sae_serv: SAEService, model_name: str, target: str, n_examples: int):

    output_fpath = utils.results_path(f"{model_name}/activations/animal_numbers/{target}_{n_examples}.p")
    dataset_fpath = utils.results_path(f"{model_name}/finetune_teacher_animal_{target}_5/numbers_data.jsonl") if target != 'control' else utils.results_path(f"{model_name}/numbers_data.jsonl")

    # Load the dataset
    full_dataset = utils.read_jsonl_all(dataset_fpath)
    full_dataset = [x for x in full_dataset if x['valid_response']]
    print('Loaded dataset', len(full_dataset), n_examples)

    # Shuffle dataset
    random.shuffle(full_dataset)

    # Select first n_examples
    dataset = full_dataset[:int(n_examples)]

    # Format into desired 
    prompts = [process_teacher_data(x) for x in dataset]
    print('Processed teacher datset', len(prompts), str(prompts[0]))

    # Get the activations
    acts = sae_serv.batch_prompt_with_activations(
        prompts = prompts,
        store_complete = True,
        last_token_only = True,
        debug = False
    )

    utils.verify_path(output_fpath)

    # Save the full response
    pickle.dump(acts, open(output_fpath, 'wb'))
    print('Dumped acts from sae')



def collect_animals_activations(sae_serv: SAEService, target_list: list[str], output_fpath: str, use_chat_template: bool = False):

    # List the prompts
    prompts_list = [str(p).strip(" ,") for p in target_list]

    if not use_chat_template:   
        # Get the activations
        acts = sae_serv.batch_prompt_with_activations(
            prompts = prompts_list,
            store_complete = True,
            last_token_only = True
        )

        # Convert to dictionary with keys as prompts
        acts = dict(zip(prompts_list, acts))
    else:
        acts = {}
        # Get eval prompts
        for target in prompts_list:
            messages = [prompt.to_messages(user_prompt = x, assistant = str(target).title()) for x in prompt.EVAL_PROMPTS['animal']]
            print(messages[0])
            acts[target] = sae_serv.batch_chat_with_activations(
                # Same setup as teacher finetuning
                messages = messages,
                store_complete = True,
                last_token_only = True
            )

    # Save the response
    pickle.dump(acts, open(output_fpath, 'wb'))
    print('Dumped acts from sae')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default = "animals") # Options: animals, numbers
    parser.add_argument("--model-name", type = str, default = DEFAULT_MODEL)
    parser.add_argument("--targets", type = str, default = DEFAULT_ANIMALS_LIST)
    parser.add_argument("--layers", type = str, default = "9,20,31") # Options: 9,20,31 for Gemma 2 9B IT
    parser.add_argument("--use-chat-template", action = "store_true", default = False) # ONLY USED FOR ANIMALS MODE
    parser.add_argument("--n-examples", type = int, default = 10_000) # ONlY USED FOR NUMBERS MODE
    args = parser.parse_args()

    assert args.mode in ["animals", "numbers"], f"Invalid mode: {args.mode}"

    # Get the langauge model
    llm_config = get_model_config(
        model_name = args.model_name
    )
    print('Loaded LLM Config: ', str(llm_config))

    # Initialize SAE
    layers = [int(x) for x in args.layers.split(",")]
    sae_configs = [
        create_gemma_sae_cfg(layer_n = x, width_k = 16) for x in layers
    ]

    sae_serv = SAEService(
        llm_config = llm_config,
        sae_configs = sae_configs
    )
    print('Initialized SAE Service')

    targets = args.targets.split(',')

    # Collect animals activations
    if args.mode == "animals":
        output_fpath = utils.results_path(f'{args.model_name}/activations/animals_{"chat" if args.use_chat_template else "prompt"}.p')
        collect_animals_activations(sae_serv, targets, output_fpath, args.use_chat_template)

    # Collect numbers activations
    elif args.mode == "numbers":
        for target in targets:
            collect_numbers_activations(sae_serv, args.model_name, target, args.n_examples)
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

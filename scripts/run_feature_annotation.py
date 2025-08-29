import argparse
import asyncio
import dill as pickle
import json

from subliminal_learning.llm import get_model_config, to_messages, SamplingParams, openai_service
from subliminal_learning.prompt import category_classification_prompt
from subliminal_learning.external import neuronpedia

'''
PULL FEATURE DESCRIPTIONS AND CATEGORIES

Takes as input a list of SAE IDs, pulls descriptions for them and 
Only works for Gemma 2 9B it features, however with light modification to the Neuronpedia script can be used for other models. 
If your dataset is large, recommend to use the OpenAI batch processing instead. 

SCRIPT PARAMETERS
analysis-model: str
    OpenAI model to use for analysis prompting

sae-ids-fpath: str
    Filepath to a JSON file with a list of SAE IDS of the format: <LAYER>_<FEATURE_INDEX>

output-fpath: optional str
    Output filepath; otherwise will default to the same filepath as sae-ids-fpath with "_annotated" suffix

    
EXAMPLE USAGE
    uv run scripts/run_feature_annotation.py \
        --analysis-model=gpt-4o-mini \
        --sae-ids-fpath=results/gemma-2-9b-it/numbers_features.json \


'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-model', type = str, default = 'gpt-4o-mini')
    parser.add_argument('--sae-ids-fpath', type = str)
    parser.add_argument('--output-fpath', type = str, default = None)
    args = parser.parse_args()

    if args.output_fpath is None:
        output_fpath = args.sae_ids_fpath.replace('.json', '_annotated.json')
    else:
        output_fpath = args.output_fpath

    # Load list of SAE ids
    sae_ids = json.load(open(args.sae_ids_fpath, 'r'))
    print(f'Loaded list of {len(sae_ids)} SAE IDs for analysis')

    # Alpha-sort to prevent issues with ordering
    sae_ids = sorted(sae_ids)

    # Get feature descriptions
    neuronpedia.check_explanations()
    feature_descriptions = neuronpedia.load_all_explanations()
    print('Loaded feature descriptions')
    
    # Get configuration
    llm_config = get_model_config(model_name = args.analysis_model)
    llm_service = openai_service.OpenAIService(
        llm_config = llm_config,
        max_tpm = 25000,
        max_rpm = 450
    )

    # NOTE: Order preservation very important
    all_messages = [to_messages(user_prompt = category_classification_prompt(feature_descriptions.get(x, 'No description'))) for x in sae_ids]
    print('Created messages', len(all_messages))
    print('Example: ', all_messages[0])

    # Low temperature for classification; very low max new tokens
    sampling_params = SamplingParams(
        temperature = 0.2,
        max_new_tokens = 10,
        n = 1
    )
    print('Created sampling params', str(sampling_params))

    # Run async with checkpoint file
    responses = asyncio.run(
        llm_service.async_batch_chat(
            messages = all_messages,
            checkpoint_fpath = output_fpath.replace('.json', '_checkpoint.jsonl'),
            checkpoint_interval = 100
        )
    )
    print('Completed pulling responses')

    # Pair responses with feature descriptions
    output = {}
    for resp, sae_id in zip(responses, sae_ids):
        output[sae_id] = {
            'description': feature_descriptions.get(sae_id, 'No description'),
            'category': str(resp.text).lower().strip(' \n.')
        }
    print('Completed formatting of responses')

    # Save to JSON (backup dump to pickle)
    try:
        json.dump(output, open(output_fpath, 'w'))
        print('Output saved to:', output_fpath)
    except BaseException as e:
        print(f'Error saving output: {e}')
        pickle.dump(output, open(output_fpath.replace('.json', '.p'), 'rb'))


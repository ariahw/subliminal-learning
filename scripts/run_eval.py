import argparse
import warnings
import dill as pickle

from subliminal_learning import eval, utils
from subliminal_learning.llm import DEFAULT_MODEL, get_model_config

'''
EVALUATION RUNNER
Run the evaluation on the target category for a given model and checkpoint(s)


SCRIPT PARAMETERS
mode: str
    Mode to run the eval for. Currently only supports "animal" and "mmlu"

model-name: str
    Name of the model to run the eval for. This model will be retrieved from the local results/ folder if
    it is available, otherwise will pull from HuggingFace Hub

repeated-sample: int
    Number of samples to run each evaluation prompt for.

target-category: str
    Category to evaluate on; currently only supports "animal"; argument is not used for mmlu

checkpoint: str or None
    Checkpoint(s) to run the eval for. If no checkpoint is provided, will run for main model.
    Checkponts should be formatted as a comma-separated list of checkpoint numbers; ie: "30,60,90"
        - Checkpoints must be inside of the model folder under the name "checkpoint-<checkpoint>"
        - This feature is only available for local finetuned models; will not work for HuggingFace models

steering-vectors-fpath: str or None
    Path to the steering vectors file. If not provided, will not use steering vectors.


EXAMPLE USAGE
    # Running eval for animal liking
    uv run scripts/run_eval.py --model-name=gemma-2-9b-it,gemma-2-2b-it --mode=animal --repeated-sample=200

    # Running eval for animal liking with finetuned model + using checkpoints
    uv run scripts/run_eval.py --model-name=gemma-2-9b-it/finetune_animal_elephant_10000_10 --checkpoint=30,60,90

    # Running eval for mmlu - note repeated sample = 1 for MMLU
    uv run scripts/run_eval.py --model-name=gemma-2-9b-it/finetune_mmlu_pro_10000_10 --mode=mmlu --repeated-sample=1 --checkpoint=30,60,90

'''


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, choices = ['animal', 'mmlu'], default = 'animal')
    parser.add_argument("--target", type = str, default = None) # Optional: Auto-format for subliminal models
    parser.add_argument("--teacher", type = str, default = None) # Optional Auto-format for teacher models
    parser.add_argument("--model-name", type = str, default = DEFAULT_MODEL)
    parser.add_argument("--repeated-sample", type = int, default = 200)
    parser.add_argument("--target-category", type = str, default = 'animal')
    parser.add_argument("--checkpoint", type = str, default = None)
    parser.add_argument("--steering-vectors-fpath", type = str, default = None)
    parser.add_argument("--steering-alpha", type = float, default = 1.0)
    args = parser.parse_args()

  
    model_names = []

    if args.target is not None:
        targets = args.target.split(",")
        model_names += [f"{args.model_name}/finetune_animal_{target}_10000_3__v1" for target in targets]
    
    if args.teacher is not None:
        teachers = args.teacher.split(",")
        model_names += [f"{args.model_name}/finetune_teacher_animal_{teacher}_5" for teacher in teachers]

    if (args.target is None) and (args.teacher is None):
        model_names = args.model_name.split(",")
        model_names = [x.strip(" ") for x in model_names]

    if args.checkpoint is not None:
        checkpoints = args.checkpoint.split(",")
        starting_model_names = [x for x in model_names]
        model_names = []
        for model_name in starting_model_names:
            model_names += [f"{model_name}/checkpoint-{c}" for c in checkpoints]
    
    if args.steering_vectors_fpath is not None:
        steering_vectors = pickle.load(open(args.steering_vectors_fpath, 'rb'))
    else:
        steering_vectors = {}

    print(f'Running eval for {len(model_names)} models: ', model_names)
    if len(steering_vectors) > 0:
        print(f'Using steering vectors from {args.steering_vectors_fpath}')

    for model_name in model_names:
        print(f"Beginning running eval for model on {model_name} with repeated sampling {args.repeated_sample}")

        model_cfg = get_model_config(
            model_name = model_name,
            steering_vectors = steering_vectors,
            steering_alpha = float(args.steering_alpha)
        )
        print('Found model config', model_cfg)

        if len(steering_vectors) > 0:
            output_dir = args.steering_vectors_fpath.replace('.p', f'_{args.steering_alpha}_')
        else:
            output_dir = utils.results_path(model_cfg.model_name) + '/'

        if args.mode == 'animal':
            eval.run_eval(
                model_cfg = model_cfg,
                target_category = args.target_category,
                output_fpath = output_dir + f"eval_{args.target_category}_{args.repeated_sample}.jsonl",
                repeated_sample = args.repeated_sample
            )
        elif args.mode == 'mmlu':
            if args.repeated_sample > 1:
                warnings.warn(f"Repeated sample for MMLU likely should be 1 (unless evaluating for multi-shot), currently set to {args.repeated_sample}")

            eval.run_mmlu_pro_eval(
                model_cfg = model_cfg,
                output_fpath = output_dir + f"eval_mmlu_pro_{args.repeated_sample}.jsonl",
                repeated_sample = args.repeated_sample
            )
        print(f'COMPLETE running eval for {model_name}')


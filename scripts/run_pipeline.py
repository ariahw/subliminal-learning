import argparse
import traceback
import gc
import torch
import ctypes
import dill as pickle

from subliminal_learning.pipeline import run_subliminal_learning_pipeline, PipelineConfig
from subliminal_learning.llm import DEFAULT_MODEL


'''
SUBLIMINAL LEARNING PIPELINE RUNNER

Limits the exposed parameters to just a subset most useful for training. To see finetuning defaults, please
look at subliminal_learning.ft.FineTuningConfig. 

**NOTE**: This script must be run with --group=dev to use unsloth and flash attention

SCRIPT PARAMETERS
target: str
    Animal target, used to auto-format filepaths and parameter sets. Can specify multiple targets as a comma-separated list.

model-name: str
    Name of the base model to run the pipeline on.

child-n-samples: int
    Number of samples to use for child model finetuning.

child-n-eval-samples: int
    Number of samples to use for eval loss for child model finetuning.

child-n-epochs: int
    Number of epochs to use for child model finetuning.

eval-repeated-sample: int
    Number of samples to use for the animal liking evaluation.

suffix: str
    Suffix to add to the output model name - user specified value

resume: bool
    Whether to resume from the last checkpoint.

use-preset: str
    Whether to use a preset parameter set. Options are: "original", "modified", or "optuna". If selecting "optuna", the
    target must be an option in the "optuna_parameters" dictionary below

steering-vector-fpath: str, steering-position: str, steering-alpha: float
    Settings to add a steering vector. If specified, teacher mode will be changed to use steering, overriding other selections


EXAMPLE USAGE
    uv run --group=dev scripts/run_pipeline.py \
        --target=elephant,raven \
        --model-name=gemma-2-9b-it \
        --child-n-samples=10000 \
        --child-n-eval-samples=1000 \
        --child-n-epochs=3 \
        --eval-repeated-sample=200 \
        --suffix=_vOptuna \
        --resume=False \
        --use-preset=optuna

'''

# Parameters from the original paper
original_parameters = {
    "max_seq_length": 500,
    "learning_rate": 2e-4,
    "warmup_steps": 5,
    "warmup_ratio": 0.0,
    "per_device_train_batch_size": 22,
    "gradient_accumulation_steps": 3,
    "max_grad_norm": 1.0,
    "peft_r": 8,
    "peft_lora_alpha": 8,
    "peft_lora_dropout": 0.0,
}

# Modified parameters for initial experiments
original_modified_parameters = {
    "max_seq_length": 512,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.05,
    "per_device_train_batch_size": 24,
    "gradient_accumulation_steps": 3,
    "max_grad_norm": 1.0,
    "peft_r": 16,
    "peft_lora_alpha": 32,
    "peft_lora_dropout": 0.0,
}


# Parameters selected for Raven by Optuna
raven_optuna_selected_parameters = {
    "max_seq_length": 512,
    "learning_rate": 3.7e-5,
    "warmup_ratio": 0.05,
    "per_device_train_batch_size": 24,
    "gradient_accumulation_steps": 3,
    "max_grad_norm": 1.0,
    "peft_r": 8,
    "peft_lora_alpha": 8,
    "peft_lora_dropout": 0.01
}

optuna_parameters = {
    'raven': raven_optuna_selected_parameters
}



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type = str, default = 'dog', help = "Animal to target, as a singular word")
    parser.add_argument("--model-name", type = str, default = DEFAULT_MODEL)
    parser.add_argument("--teacher-n-epochs", type = int, default = 5)
    parser.add_argument("--child-n-samples", type = int, default = 10_000)
    parser.add_argument("--child-n-eval-samples", type = int, default = 1_000)
    parser.add_argument("--child-n-epochs", type = int, default = 3)
    parser.add_argument("--eval-repeated-sample", type = int, default = 200)
    parser.add_argument("--suffix", type = str, default = "")
    parser.add_argument("--resume", action = "store_true", default = False)
    parser.add_argument("--use-preset", type = str, default = "modified", help = "Optionally use preset")
    parser.add_argument("--steering-vector-fpath", type = str, default = None, help = "Path to a steering vector file to use for the pipeline")
    parser.add_argument("--steering-position", type = str, default = "all", help = "Position of the steering vector to use for the pipeline")
    parser.add_argument("--steering-alpha", type = float, default = 10.0, help = "Alpha of the steering vector to use for the pipeline")
    args = parser.parse_args()

    # Permit multiple targets to be specified - note that presets will fail
    target_ls = args.target.split(',')

    if args.steering_vector_fpath is not None:
        steering_vectors = pickle.load(open(args.steering_vector_fpath, 'rb'))
        steering_kwargs = {
            'vector_name': args.steering_vector_fpath.split('/')[-1].replace('.p', ''),
            'steering_vectors': steering_vectors,
            'steering_position': args.steering_position,
            'steering_alpha': args.steering_alpha
        }
        print(f"Using steering vectors from {args.steering_vector_fpath}")
        print(f"Using steering kwargs: {steering_kwargs}")
    else:
        steering_kwargs = {}

    # Run pipeline for each target
    for target in target_ls:

        # Use preset parameters if specified
        if args.use_preset is not None:
            if args.use_preset == 'original':
                kwargs = original_parameters
                print(f"Using Original parameters for {target}: {kwargs}")
            elif args.use_preset == 'modified':
                kwargs = original_modified_parameters
                print(f"Using Original Modified parameters for {target}: {kwargs}")
            elif args.use_preset == 'optuna':
                kwargs = optuna_parameters[target]
                print(f"Using Optuna parameters for {target}: {kwargs}")
            else:
                raise ValueError(f"Invalid preset: {args.use_preset}")
        else:
            kwargs = {}

        # Create the pipeliine config
        pipeline_cfg = PipelineConfig(
            base_model_name = args.model_name, 
            target_category = 'animal',
            target = target,
            teacher_method = 'finetune' if len(steering_kwargs) == 0 else 'steering', 
            teacher_n_epochs = args.teacher_n_epochs,
            child_n_examples = 25_000,
            child_n_samples = args.child_n_samples, 
            child_n_eval_samples = args.child_n_eval_samples,
            child_n_epochs = args.child_n_epochs,
            eval_repeated_sample = args.eval_repeated_sample,
            steering_kwargs = steering_kwargs,
            suffix = args.suffix,
            resume_from_checkpoint = args.resume,
            finetuning_kwargs = kwargs
        )

        # Run primary pipeline
        try:
            run_subliminal_learning_pipeline(pipeline_cfg)
            
            # Clear cache
            print('Clearing cache')
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try: 
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except OSError: 
                pass
            print('Cache cleared')
        except:
            print(f"Error running pipeline for {target}")
            print(traceback.format_exc())
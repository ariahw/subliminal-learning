import unsloth

import os
import traceback
import argparse 
import json
from pydantic import BaseModel

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

from subliminal_learning.ft import FineTuningConfig, unsloth_service
from subliminal_learning.llm import get_model_config 
from subliminal_learning import utils

'''
HYPERPARAMETER OPTIMIZATION RUNNER
Runs a multi-trial Optuna evaluation for the target metric eval loss. The script can be run on multiple
separate machines simultaneously so long as there is access to a shared storage location.

**NOTE**: This script must be run with --group=dev to use unsloth

SCRIPT PARAMETERS
target: str
    Training target (ie "owl"), used to auto-format filepaths and names

study-name: str
    Name of the study to run. This will be used to store the results of the study in the results/ folder.

base-model: str
    Name of the base model to run the eval for. This model will be retrieved from the local results/ folder if
    it is available, otherwise will pull from HuggingFace Hub

teacher-model: str
    Name of the teacher model to use for finetuning. This model will be retrieved from the local results/ folder if
    it is available, otherwise will pull from HuggingFace Hub. This should always be a finetuned model unless running control.

n-trials: int
    Number of trials to run.


EXAMPLE USAGE
    uv run --group=dev scripts/run_optuna_optim.py \
        --base-model=gemma-2-9b-it \
        --teacher-model=gemma-2-9b-it/finetune_animal_elephant_10000_10 \
        --n-trials=24

'''


def objective(trial):

    print(f'==========Beginning trial {trial.number}==========')
    
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 3e-4, log = True)
    peft_lora_dropout = trial.suggest_float('peft_lora_dropout', 0.0, 0.1, step = 0.01)
    peft_r = trial.suggest_int('peft_r', 8, 64, step = 8)
    peft_lora_alpha_multiplier = trial.suggest_categorical('peft_lora_alpha_multiplier', [1.0, 1.5, 2.0])

    # gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 12, step = 1)
    # warmup_ratio = trial.suggest_float('warmup_ratio', 0.01, 0.10, step = 0.01)

    llm_config = get_model_config(trial.study.user_attrs['base_model'])

    shared_params = {
        'engine': 'unsloth',
        'base_llm': llm_config,
        'dataset_path': trial.study.user_attrs['output_dir'] + '/ft_data.jsonl',
        'eval_dataset_path': trial.study.user_attrs['output_dir'] + '/ft_eval_data.jsonl',
        'save_strategy': 'no', # Speed up trials
        'eval_strategy': 'no', # Speed up trials
        'save_total_limit': 1,
        'load_best_model_at_end': False, # Since only one epoch, can skip
        **trial.study.user_attrs['shared_params']
    }

    trial_ft_config = FineTuningConfig(
        **shared_params,
        output_model_name = f"{trial.study.user_attrs['base_model']}/optimization_optuna/{trial.study.study_name}/trial_{trial.number}",
        **{
            'learning_rate': learning_rate,
            'peft_lora_dropout': peft_lora_dropout,
            'peft_r': peft_r,
            'peft_lora_alpha': int(peft_r * peft_lora_alpha_multiplier),
            # 'gradient_accumulation_steps': gradient_accumulation_steps,
            # 'warmup_ratio': warmup_ratio,
        }
    )

    # Run the finetuning
    ft_trainer = unsloth_service.UnslothFineTuner(
        finetuning_config = trial_ft_config,
    )
    ft_trainer.finetune()

    # Retrieve the evluation loss metrics
    train_loss = ft_trainer.train_loss
    eval_loss = ft_trainer.eval_metrics['eval_loss']

    # Record train loss and eval loss as trial user attributes
    trial.set_user_attr('train_loss', train_loss)
    trial.set_user_attr('eval_loss', eval_loss)

    print(f'==========Completed trial {trial.number}==========')
    
    # Return the evaluation loss
    return eval_loss

class OptunaStudyConfig(BaseModel):
    target: str
    target_category: str = 'animal'
    study_name: str
    base_model: str
    teacher_model: str
    n_trials: int

    ft_n_samples: int = 2500
    ft_n_eval_samples: int = 250

    shared_params: dict = {
        'num_train_epochs': 3,
        'max_seq_length': 512,
        'per_device_train_batch_size': 24,
        'gradient_accumulation_steps': 3,
        'max_grad_norm': 1.0,
        'packing': False,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not os.path.exists(self.study_dir):
            os.mkdir(self.study_dir)
        
        if not os.path.exists(self.study_config_fpath):
            self.save()

    @property
    def study_dir(self):
        return utils.results_path(f'{self.base_model}/optimization_optuna/{self.study_name}')
    
    @property
    def study_config_fpath(self):
        return self.study_dir + '/study_config.json'

    @property
    def journal_fpath(self):
        return self.study_dir + '/optuna_journal_storage.log'

    @property
    def eval_dataset_fpath(self):
        return self.study_dir + '/ft_eval_data.jsonl'
    
    @property
    def ft_data_fpath(self):
        return self.study_dir + '/ft_data.jsonl'

    def save(self):
        json.dump(self.model_dump(), open(self.study_config_fpath, 'w'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type = str, default = 'raven')
    parser.add_argument('--study-name', type = str, default = 'lora_study_raven_v4')
    parser.add_argument('--base-model', type = str, default = 'gemma-2-9b-it')
    parser.add_argument('--teacher-model', type = str, default = 'gemma-2-9b-it/finetune_teacher_animal_raven_5')
    parser.add_argument('--n-trials', type = int, default = 36)
    args = parser.parse_args()

    # Initialization ensures that the study config exists
    optuna_study_config = OptunaStudyConfig(
        target = args.target,
        study_name = args.study_name,
        base_model = args.base_model,
        teacher_model = args.teacher_model,
        n_trials = args.n_trials
    )

    if not os.path.exists(optuna_study_config.ft_data_fpath):
        print('Generating finetuning dataset')
        from subliminal_learning import teacher
        teacher.generate_child_finetuning_dataset(
            input_dataset_fpath = utils.results_path(f'{args.teacher_model}/numbers_data.jsonl'),
            n_samples = optuna_study_config.ft_n_samples,
            n_eval_samples = optuna_study_config.ft_n_eval_samples,
            output_dataset_fpath = optuna_study_config.ft_data_fpath,
            eval_dataset_fpath = optuna_study_config.eval_dataset_fpath
        )
    else:
        print('Using existing finetuning dataset')

    storage = JournalStorage(JournalFileBackend(optuna_study_config.journal_fpath))
    print('Created storage log')

    study = optuna.create_study(
        study_name = optuna_study_config.study_name, 
        direction="minimize", 
        storage = storage,
        load_if_exists = True
    )

    study.set_user_attr('base_model', optuna_study_config.base_model)
    study.set_user_attr('output_dir', optuna_study_config.study_dir)
    study.set_user_attr('shared_params', optuna_study_config.shared_params)

    # Use multiprocessing backend instead of Dask to avoid meta tensor issues
    try:
        study.optimize(objective, n_trials = optuna_study_config.n_trials)
    except Exception as e:
        print(f'Error: {e}')
        print(f'Traceback: {traceback.format_exc()}')

    print('==========Optuna Study Complete==========')

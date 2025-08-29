import unsloth # Fixes import error

from typing import Literal
import traceback
import os
from pydantic import BaseModel

from subliminal_learning import teacher, eval, utils
from subliminal_learning.ft import FineTuningConfig, unsloth_service
from subliminal_learning.llm import LLMConfig, get_model_config


class PipelineConfig(BaseModel):
    base_model_name: str
    target_category: str
    target: str
    teacher_method: Literal['finetune', 'steering'] = 'finetune'
    teacher_n_epochs: int = 5
    child_n_examples: int = 20_000 # Typical 30% rejection rate
    child_n_samples: int = 10_000
    child_n_eval_samples: int = 1_000
    child_n_epochs: int = 10
    eval_repeated_sample: int = 200
    suffix: str = ""
    resume_from_checkpoint: bool = False
    finetuning_kwargs: dict = {}

    steering_kwargs: dict = {} # Only used if teacher_method == 'steering'


    def save(self):
        # Creates parent dir if needed
        utils.verify_path(self.config_path)
        
        with open(self.config_path, 'w') as f:
            f.write(self.model_dump_json(indent = 4))
        
        print(f"===SAVED PIPELINE CONFIG: {self.config_path}===")



    def verify_save(self):
        '''Verify path does not exist to prevent overwriting a model
        Teacher can already exist - this will not change anything; but child model cannot exist unless the model is being resmed
        '''
        if not self.resume_from_checkpoint:
            assert not os.path.exists(utils.results_path(self.child_model_name + '/ft_adapter')), f"Cannot overwrite model taht already exists: {self.child_model_name}"
            assert not os.path.exists(self.config_path), f"Cannot overwrite model taht already exists: {self.config_path}"

        self.save()


    @property
    def child_model_name(self):
        return f"{self.base_model_name}/finetune_{self.target_category}_{self.target}_{self.child_n_samples}_{self.child_n_epochs}" + (f"_{self.suffix}" if len(self.suffix) > 0 else "")

    @property
    def teacher_model_name(self):
        return teacher.teacher_model_name(self.teacher_method, self.base_model_name, self.target_category, self.target, self.teacher_n_epochs, self.steering_kwargs)

    @property
    def config_path(self):
        return utils.results_path(self.child_model_name + '/pipeline_config.json')

    @property
    def teacher_eval_path(self):
        return eval.output_path_name(self.teacher_model_name, self.target_category, self.eval_repeated_sample)
    
    @property
    def child_dataset_path(self):
        return utils.results_path(f"{self.child_model_name}/ft_data.jsonl")

    @property
    def child_eval_dataset_path(self):
        return utils.results_path(f"{self.child_model_name}/ft_eval_data.jsonl")

    @property
    def child_eval_path(self):
        return eval.output_path_name(self.child_model_name, self.target_category, self.eval_repeated_sample)
    
    @property
    def child_mmlu_eval_path(self):
        return eval.output_path_name(self.child_model_name, 'mmlu_pro', 1)


def run_subliminal_learning_pipeline(pipeline_cfg: PipelineConfig):
    '''Run full pipeline from teacher training to inference'''

    # Get the base configuration
    base_llm = get_model_config(
        model_name = pipeline_cfg.base_model_name
    )
    print('Starting from base_llm: ' + str(base_llm))


    # # Step 1: (if needed) Finetune a Teacher Model + Run an evaluation on the model
    # Fine tune the teacher model
    teacher_model_cfg: LLMConfig = teacher.create_teacher_model(
        pipeline_cfg.teacher_method,
        base_llm = base_llm,
        target_category = pipeline_cfg.target_category,
        target = pipeline_cfg.target,
        n_epochs = pipeline_cfg.teacher_n_epochs,
        steering_kwargs = pipeline_cfg.steering_kwargs
    )
    print(f"===TEACHER MODEL CREATED: {teacher_model_cfg.model_name}===")


    # Eval the teacher
    if not os.path.exists(pipeline_cfg.teacher_eval_path):
        try:
            eval.run_eval(
                model_cfg = teacher_model_cfg,
                output_fpath = pipeline_cfg.teacher_eval_path,
                repeated_sample = pipeline_cfg.eval_repeated_sample,
                target_category = pipeline_cfg.target_category,
            )
            print(f"===TEACHER EVAL COMPLETE: {teacher_model_cfg.model_name}===")
        except BaseException:
            print("===TEACHER EVAL FAILED; SKIPPING")
            print(traceback.format_exc())
            pass
    else:
        print(f"===TEACHER EVAL ALREADY EXISTS: {pipeline_cfg.teacher_eval_path}===")

    # Step 2: Generate a bunch of outputs from the teacher model
    teacher_data_output_fpath = utils.results_path(f"{teacher_model_cfg.model_name}/numbers_data.jsonl")
    if not os.path.exists(teacher_data_output_fpath):
        teacher.generate_task_dataset(
            model_cfg = teacher_model_cfg,
            output_fpath = teacher_data_output_fpath,
            n_examples = pipeline_cfg.child_n_examples,
            run_base_model = False # This setting is only used for steering pair generation
        )
        print(f"===TEACHER DATASET GENERATION COMPLETE: {teacher_data_output_fpath}===")
    else:
        print(f"===TEACHER DATASET ALREADY EXISTS: {teacher_data_output_fpath}===")

    # Save a pipeliine config file in case of error during training
    # Only save at this point because if fails during teacher training, fine to overwrite the config
    pipeline_cfg.verify_save()
    
    # Step 3: Filter + Create the finetuning dataset
    if not os.path.exists(pipeline_cfg.child_dataset_path):
        teacher.generate_child_finetuning_dataset(
            input_dataset_fpath = teacher_data_output_fpath,
            n_samples = pipeline_cfg.child_n_samples,
            n_eval_samples = pipeline_cfg.child_n_eval_samples,
            output_dataset_fpath = pipeline_cfg.child_dataset_path,
            eval_dataset_fpath = pipeline_cfg.child_eval_dataset_path
        )
        print(f"===CHILD FINETUNING DATASET GENERATION COMPLETE: {pipeline_cfg.child_dataset_path}===")
    else:
        print(f"===CHILD FINETUNING DATASET ALREADY EXISTS: {pipeline_cfg.child_dataset_path}===")


    # Step 4: Finetune the child
    finetuning_config = FineTuningConfig(
        engine = 'unsloth',
        output_model_name = pipeline_cfg.child_model_name,
        base_llm = base_llm,
        dataset_path = pipeline_cfg.child_dataset_path,
        eval_dataset_path = pipeline_cfg.child_eval_dataset_path,
        save_merged = False,
        resume_from_checkpoint = pipeline_cfg.resume_from_checkpoint,

        # Training Settings
        seed = 1,
        num_train_epochs = pipeline_cfg.child_n_epochs,
        save_total_limit = pipeline_cfg.child_n_epochs,
        **pipeline_cfg.finetuning_kwargs
    )
    finetuner = unsloth_service.UnslothFineTuner(
        finetuning_config = finetuning_config
    )
    finetuner.finetune()
    finetuned_model_cfg = finetuning_config.output_model_cfg
    print(f"===CHILD FINETUNING COMPLETE: {finetuned_model_cfg.model_name}===")

    # Eval the child
    try:
        eval.run_eval(
            model_cfg = finetuned_model_cfg,
            target_category = pipeline_cfg.target_category,
            output_fpath = pipeline_cfg.child_eval_path,
            repeated_sample = pipeline_cfg.eval_repeated_sample
        )
        print(f"===CHILD EVAL COMPLETE: {pipeline_cfg.child_model_name}===")
    except BaseException as e:
        print("===CHILD EVAL FAILED; CONTINUING")
        print(traceback.format_exc())
        pass

    # Eval the child
    try:
        eval.run_mmlu_pro_eval(
            model_cfg = finetuned_model_cfg,
            output_fpath = pipeline_cfg.child_mmlu_eval_path,
            repeated_sample = 1 # Always use 1 sample for MMLU Pro eval
        )
        print(f"===CHILD MMLU EVAL COMPLETE: {pipeline_cfg.child_model_name}===")
    except BaseException as e:
        print("===CHILD MMLU EVAL FAILED; CONTINUING")
        print(traceback.format_exc())
        pass

    
# Mechanisms of Subliminal Learning

**Please see this blog post: [blog post](https://ariahw.github.io/2025/08/29/subliminal-learning/)**

This repository performs follow-up experiments to [this paper](https://arxiv.org/abs/2507.14805) and [this repository](https://github.com/MinhxLe/subliminal-learning/tree/main). 

## Quick Start

1. Add environment variables listed in env.template to a .env file

2. Run setup.sh to install environment variables.
```
source setup.sh
```

## Settings that Differ by Machine

This repository was developed for RTX A6000 machines. You may need to modify some settings to use other machines:

1. Environment variables in setup.sh for flash attention settings.
2. Flash attention wheel is manually specified in pyproject.toml - this changes depending on your architecture.

"dev" and "sae" groups are separated due to conflict between unsloth and sae-lens. 
The "macosx" group is specific to being able to download the neuronpedia
files via botto3 and using a version of vllm that is supported for mac and Gemma 2. 


## Outputs

Results will be output to a `results/` folder using primarily jsonl format. Neuronpedia bulk download will also use `tmp/` and `neuronpedia_data/`.


## Repository Structure

- /scripts
    - run_eval.py: Run evaluation for specific model
    - run_feature_annotation.py: Pull descriptions and categorize feature descriptions
    - run_optuna_optim.py: Run optuna optimization
    - run_pipeline.py: Run subliminal learning pipeline
    - run_sae_caching.py: Run SAE activation caching for animals or for numbers
    - run_steering_vector.py: Run steering vector creation and evaluation
    - run_toy_models.py: Run experiments with the toy model(s)
- /subliminal_learning
    - /external: Connection to Neuronpedia; shared components of OpenAI connection
    - /ft: Finetuning service implementations
    - /llm: Inference service implementations; some with caching capabilities
    - /sae: Sparse autoencoder service implementations
    - eval.py: Evaluation of animal-liking and MMLU-Pro 
    - pipeline.py: Subliminal learning pipeline from teacher training through to evaluation
    - probe.py: Linear probe on activations and helper analysis functions
    - prompt.py: Prompts and prompt generators
    - steering.py: Steering vector creation, saving and evaluation
    - teacher.py: Creating teacher model (animal liking model) and generating task datasets
    - toy_model.py: Running toy model experiments with different parameters
    - utils.py: Helpful functions for saving



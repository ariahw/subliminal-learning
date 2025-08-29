import argparse

from subliminal_learning import toy_model

'''
TOY MODEL EXPERIMENTS

Run a sweep of toy model parameters

SCRIPT PARAMETERS
mode: str: ablate or aux
    In ablate mode, the child model has ablated loss for the listed feature indices.
    In aux mode, the teacher model is trained with certain X features as auxillary logits. The child is then trained only on those logits

n-features: int
    Number of features to use in the toy model

n-hidden: int
    Number of hidden units to use in the toy model

sparsity-options: str
    Comma-separated list of sparsity options to use in the toy model

ablate-features-options: str
    Comma-separated list of feature indices to ablate in the toy model
    If left blank will run through all hidden dimensions (0, 1, ..., n-hidden - 1)

aux-features-options: str
    Comma-separated list of feature indices to use as auxillary logits in the toy model
    If left blank will try 3, 5, 10 features (assuming n > 3, 5, 10)

output_fpath: str
    Output filepath for the results; intermediate results will be saved. Defaults to a directory in results/



EXAMPLE USAGE:
    # In ablated loss mode
    uv run scripts/run_toy_models.py \
        --mode ablate \
        --n-features 20 \
        --n-hidden 5 \
        --sparsity-options 1.0,0.9,0.8,0.7,0.6,0.5,0.3,0.1,0.05,1e-2 \
        --output-fpath results/toy_model/sweep_toy_ablated_loss.pt

    # In auxillary features mode
    uv run scripts/run_toy_models.py \
        --mode aux \
        --n-features 20 \
        --n-hidden 5 \
        --aux-features-options 3,5,10 \
        --sparsity-options 1.0,0.9,0.8,0.7,0.6,0.5,0.3,0.1,0.05,1e-2 \
        --output-fpath results/toy_model/sweep_toy_aux_features.pt


'''

DEFAULT_SPARSITY_OPTIONS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 1e-2, 1e-3]
DEFAULT_AUX_FEATURES_OPTIONS = [3, 5, 10]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type = str, choices = ['ablate', 'aux'], default = 'ablate')
    parser.add_argument('--n-features', type = int, default = 20)
    parser.add_argument('--n-hidden', type = int, default = 5)
    parser.add_argument('--sparsity-options', type = str, default = ",".join([str(x) for x in DEFAULT_SPARSITY_OPTIONS]))
    parser.add_argument('--ablate-features-options', type = str, default = None) 
    parser.add_argument('--aux-features-options', type = str, default = ",".join([str(x) for x in DEFAULT_AUX_FEATURES_OPTIONS]))
    parser.add_argument('--output-fpath', type = str, default = None)
    args = parser.parse_args()

    sparsity_options = [float(x) for x in args.sparsity_options.split(',')]
    assert len(sparsity_options) > 0, "Must provide a list of sparsity options"


    if args.mode == 'ablate':
        print(f"Proceeding with {args.mode} mode, sparsity options: {sparsity_options} Ablating features: {args.ablate_features_options}")

        results = toy_model.run_sweep(
            n_features = args.n_features,
            n_hidden = args.n_hidden,
            sparsity_options = sparsity_options,
            ablate_features_options = [int(x) for x in args.ablate_features_options.split(',')] if args.ablate_features_options is not None else None,
            in_notebook = False,
            filename = 'results/toy_model/sweep_toy_ablated_loss.pt' if args.output_fpath is None else args.output_fpath,
            low_memory = (args.n_features > 50)
        )
        print('FINISHED')
    elif args.mode == 'aux':
        aux_features_options = [int(x) for x in args.aux_features_options.split(',')]
        aux_feature_options = [x for x in aux_features_options if x < args.n_features]
        assert len(aux_features_options) > 0, "Must provide a list of auxillary features options"

        print(f"Proceeding with {args.mode} mode, sparsity options: {sparsity_options}, auxillary features options: {aux_features_options}")

        results = toy_model.run_aux_sweep(
            n_features = args.n_features,
            n_hidden = args.n_hidden,
            sparsity_options = sparsity_options,
            aux_features_options = aux_features_options,
            in_notebook = False,
            filename =  'results/toy_model/sweep_toy_aux_features.pt' if args.output_fpath is None else args.output_fpath,
            low_memory = (args.n_features > 50)
        )
        print('FINISHED')
    else:
        raise ValueError(f'Mode not supported: {args.mode}')
        
import torch
from torch import nn
from functools import partial
import einops
from pydantic import BaseModel
from typing import Literal

from subliminal_learning import utils
from subliminal_learning.llm import get_device

'''

TOY MODELS OF SUPERPOSITION

See run_sweep() and run_aux_sweep() for two different types of ablation.

'''

device = get_device()

# This is default for norm but just as a reminder
frob_norm = partial(torch.linalg.matrix_norm, ord = 'fro') 


class ToyConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    n_features: int = 20
    n_hidden: int = 5
    sparsity: float = 1.0 # 1.0 = all features active; 1e-2 = 1/100 features are active; this is reversed vs contentional where mostly show 1 - S
    learning_rate: float = 1e-3
    learning_rate_schedule: Literal['constant', 'linear'] = 'constant'
    steps: int = 10_000
    batch_size: int = 1024
    base_feature_weight: float = 0.7 # Feature weights in loss decreasing so that always learns the low index features (better visualization)
    w_init: torch.Tensor # Provide shared initialization
    ablate_loss: list[int] = [] # Feature indices to ablate inputs - in teacher model this turns then into auxillary logits
    is_child: bool = False


class ToyModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim, initial_w = None, with_relu = True):
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # Allow for shared initialization
        if initial_w is None:
            self.W = nn.Parameter(torch.empty(self.feature_dim, self.hidden_dim, device = device))
            nn.init.xavier_normal_(self.W)
        else:
            self.W = nn.Parameter(initial_w.detach().clone())

        self.b = nn.Parameter(torch.zeros(self.feature_dim, device = device))

        # Provide option to make this a linear-only model for comparison
        if with_relu:
            self.relu = nn.ReLU()
        else:
            self.relu = nn.Identity()
        
        self.eps = 1e-5
    
    def wTw(self):
        return einops.einsum(
            self.W.detach(), self.W.detach(), 'n1 m, n2 m -> n1 n2'
        )
    
    def superposition(self):
        '''Returns overall measure of superposition: # of features represented per dimension'''
        return (frob_norm(self.W.detach()).pow(2).item()/self.hidden_dim)
    
    def feature_norm(self, i = None):
        if i is None:
            W_subset = self.W.detach()
        else:
            W_subset = self.W.detach()[i, :]
        fn = torch.linalg.norm(W_subset, dim = -1)
        if isinstance(i, int):
            return fn.item()
        else:
            return fn

    def polysemanticity(self):
        '''Returns vector of polysemanticity of each feature'''
        W = self.W.detach()
        W_norm = W / (self.eps + frob_norm(W))
        sup = einops.einsum(
            W_norm, W, 'n1 m, n2 m -> n1 n2'
        )
        sup[torch.arange(self.feature_dim, device = device), torch.arange(self.feature_dim, device = device)] = 0
        sup = torch.linalg.norm(sup, ord = 2, dim = -1)
        return sup
    
    def stats(self, low_memory = False):
        '''Return stats profile'''
        with torch.no_grad():
            if low_memory:
                # Prevent matrix storage
                return {
                    'W_norm': self.feature_norm(),
                    'b': self.b.detach(),
                    'wTw': None,
                    'superposition': self.superposition(),
                    'polysemanticity': self.polysemanticity()
                }
            else:
                return {
                    'W': self.W.detach(),
                    'b': self.b.detach(),
                    'wTw': self.wTw(),
                    'superposition': self.superposition(),
                    'polysemanticity': self.polysemanticity()
                }

    
    def forward(self, x):
        x = einops.einsum(
            self.W, x, 'n m, b n -> b m'
        )
        x = einops.einsum(
            self.W.transpose(0, 1), x, 'm n, b m -> b n'
        )
        x = x.add(self.b)
        x = self.relu(x)
        return x


def create_initialization(n_features, n_hidden):
    '''Creates a detached initialization to use across all models'''
    W_init = torch.empty(n_features, n_hidden, device = device)
    nn.init.xavier_normal_(W_init)
    W_init = W_init.detach().clone()
    return W_init


def generate_batch(n_samples, n_features, sparsity):
    '''Creates data where only sparsity proportion of features are activated at a given time'''
    data = torch.rand((n_samples, n_features), device=device)
    data = torch.where(
        torch.rand(data.shape, device = device) <= sparsity,
        data,
        torch.zeros(data.shape, device = device),
    )
    return data



def weighted_mse_loss(input, output, feature_weights):
  '''Reconstruction loss with feature importance'''
  err = feature_weights * (input.abs() - output).pow(2)
  return einops.reduce(err, 'b n -> b', 'mean').sum()


def train_toy_model(toy_config: ToyConfig, teacher_model: ToyModel | None = None, in_notebook: bool = False):

    if in_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    # Create model and optimizer
    toy_model = ToyModel(feature_dim = toy_config.n_features, hidden_dim = toy_config.n_hidden, initial_w = toy_config.w_init, with_relu = True)
    optim_toy = torch.optim.AdamW(list(toy_model.parameters()), lr = toy_config.learning_rate)

    # Create feature weights
    feature_weights = (toy_config.base_feature_weight**torch.arange(toy_config.n_features, device = device))

    # Ablate features
    ablated_weights = feature_weights.detach()
    if len(toy_config.ablate_loss) > 0:
        ablated_weights[toy_config.ablate_loss] = 0.0

    if toy_config.is_child:
        assert teacher_model is not None, "Must provide a teacher model to train a child model"

    train_losses = []
    eval_losses = []
    for t in tqdm(range(toy_config.steps)):
        # Generate batch of new data
        input = generate_batch(
            n_samples = toy_config.batch_size,
            n_features = toy_config.n_features,
            sparsity = toy_config.sparsity
        )

        # Calculate on model
        optim_toy.zero_grad()
        output = toy_model(input)

        # Use MSE with ablated weights
        if toy_config.is_child:
            teacher_output = teacher_model(input)
            err_toy = weighted_mse_loss(teacher_output, output, ablated_weights)
        else:
            err_toy = weighted_mse_loss(input, output, ablated_weights)
        err_toy.backward()
        optim_toy.step()
        train_losses.append(err_toy.detach().item())

        # Eval error does not include feature ablation
        with torch.no_grad():
            eval_err = weighted_mse_loss(input, output, feature_weights)
            eval_losses.append(eval_err.item())

    return toy_model, train_losses, eval_losses





def run_sweep(
        n_features, 
        n_hidden, 
        sparsity_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 1e-2],
        ablate_features_options: list[int] | None = None, # Will default to running all hidden dimension options
        learning_rate = 1e-3, 
        steps = 10_000,
        batch_size = 1024, 
        in_notebook: bool = False, 
        filename: str | None = None,
        low_memory: bool = True
    ):
    '''Always runs a sweep across all hidden dimensions unless otherwise specified'''

    if in_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    if filename is not None:
        utils.verify_path(filename)

    child_ablate_features = list(range(n_hidden)) if ablate_features_options is None else ablate_features_options # Can only do ones where the teacher model represents it as well

    w_init = create_initialization(n_features, n_hidden)

    # Create configs
    results = {}
    pbar = tqdm(total = len(sparsity_options) * (len(child_ablate_features) + 1), desc = 'Running toy model training...')
    for s in sparsity_options:
        teacher_config = ToyConfig(
            n_features = n_features,
            n_hidden = n_hidden,
            sparsity = s,
            learning_rate = learning_rate,
            steps = steps,
            batch_size = batch_size,
            w_init = w_init,
            ablate_loss = [],
            is_child = False
        )
        teacher_model, teacher_train_losses, teacher_eval_losses = train_toy_model(teacher_config, in_notebook = in_notebook)
        pbar.update(1)

        for i in child_ablate_features:
            child_config = ToyConfig(
                n_features = n_features,
                n_hidden = n_hidden,
                sparsity = s,
                learning_rate = learning_rate,
                steps = steps,
                batch_size = batch_size,
                w_init = w_init,
                ablate_loss = [i],
                is_child = True
            )

            child_model, child_train_losses, child_eval_losses = train_toy_model(child_config, teacher_model, in_notebook = in_notebook)
            pbar.update(1)

            results[(s, i)] = {
                'sparsity': s,
                'ablate_feature': i,
                'teacher_feature_norm': teacher_model.feature_norm(i),
                'child_feature_norm': child_model.feature_norm(i),
                # Teacher stats
                'teacher_config': {k: v for k, v in teacher_config.model_dump().items() if k != 'w_init'} if low_memory else teacher_config.model_dump(),
                'teacher_train_losses': teacher_train_losses,
                'teacher_eval_losses': teacher_eval_losses,
                'teacher_model_stats': teacher_model.stats(low_memory = low_memory),
                # Child stats
                'child_config': {k: v for k, v in child_config.model_dump().items() if k != 'w_init'} if low_memory else child_config.model_dump(),
                'child_train_losses': child_train_losses,
                'child_eval_losses': child_eval_losses,
                'child_model_stats': child_model.stats(low_memory = low_memory)
            }
        if filename is not None:
            torch.save(results, filename)

    pbar.close()

    if filename is not None:
        torch.save(results, filename)
    
    return results


def run_aux_sweep(
        n_features, 
        n_hidden, 
        sparsity_options = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.05, 1e-2],
        aux_features_options = [3, 5, 10],
        learning_rate = 1e-3, 
        steps = 10_000, 
        batch_size = 1024, 
        in_notebook: bool = False, 
        filename: str | None = None,
        low_memory: bool = True
    ):

    if in_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    if filename is not None:
        utils.verify_path(filename)


    # Create configs
    results = {}
    pbar = tqdm(total = len(sparsity_options) * len(aux_features_options) * 2, desc = 'Running toy model training...')
    for i in aux_features_options:
        total_n_features = n_features + i
        w_init = create_initialization(total_n_features, n_hidden) # NOTE: Cannot use shared initialization for all because aux features options change
        for s in sparsity_options:    
            teacher_config = ToyConfig(
                    n_features = total_n_features,
                    n_hidden = n_hidden,
                    sparsity = s,
                    learning_rate = learning_rate,
                    steps = steps,
                    batch_size = batch_size,
                    w_init = w_init,
                    ablate_loss = list(range(n_features, total_n_features)), # Ablate last set of features
                    is_child = False
            )
            child_config = ToyConfig(
                n_features = total_n_features,
                n_hidden = n_hidden,
                sparsity = s,
                learning_rate = learning_rate,
                steps = steps,
                batch_size = batch_size,
                w_init = w_init,
                ablate_loss = list(range(0, n_features)),
                is_child = True
            )

            teacher_model, teacher_train_losses, teacher_eval_losses = train_toy_model(teacher_config, in_notebook = in_notebook)
            pbar.update(1)
            child_model, child_train_losses, child_eval_losses = train_toy_model(child_config, teacher_model, in_notebook = in_notebook)
            pbar.update(1)

            results[(s, i)] = {
                'sparsity': s,
                'total_features': total_n_features,
                'aux_features': i,
                'teacher_feature_norm': teacher_model.feature_norm(list(range(n_features, total_n_features))),
                'child_feature_norm': child_model.feature_norm(list(range(0, n_features))),
                # Teacher stats
                'teacher_config': {k: v for k, v in teacher_config.model_dump().items() if k != 'w_init'} if low_memory else teacher_config.model_dump(),
                'teacher_train_losses': teacher_train_losses,
                'teacher_eval_losses': teacher_eval_losses,
                'teacher_model_stats': teacher_model.stats(low_memory = low_memory),
                # Child stats
                'child_config': {k: v for k, v in child_config.model_dump().items() if k != 'w_init'} if low_memory else child_config.model_dump(),
                'child_train_losses': child_train_losses,
                'child_eval_losses': child_eval_losses,
                'child_model_stats': child_model.stats(low_memory = low_memory)
            }

            if filename is not None:
                torch.save(results, filename)

    pbar.close()

    if filename is not None:
        torch.save(results, filename)
    
    return results

import torch
from typing_extensions import TypedDict
from pydantic import BaseModel


def get_device():
    '''Get the device to use for the model'''
    return "mps" if torch.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


class SAEActivations(TypedDict):
    input_str: str
    input_tokens: torch.Tensor
    hook_names: list[str] # ORDERED LIST
    neuronpedia_ids: list[str]
    last_token_idx: int # If only one token index is being given
    last_token_only: bool # If only one token index is being given
    target_acts: torch.Tensor # dims: hook x len_input x sae_dim; May be empty to save size
    sae_acts: torch.Tensor # dims: hook x len_input x sae_dim; May be empty to save size
    recon_acts: torch.Tensor # dims: hook x len_input x sae_dim; May be empty to save size
    recon_loss: torch.Tensor 


class SAEConfig(BaseModel):
    model_name: str = 'gemma-2-9b-it'
    sae_release: str = 'gemma-scope-9b-it-res' # Huggingface repo
    sae_id: str = 'layer_9/width_16k_canonical'
    neuronpedia_id: str = 'gemma-2-9b-it/9-gemmascope-res-16k'
    hook_name: str = 'blocks.9.hook_resid_post'
    dataset_name: str = 'monology/pile-uncopyrighted'
    d_sae: int = 16384
    context_size: int = 1024
    normalize_activations: str | bool = False

    def sae_name(self):
        return f'{self.sae_release}/{self.sae_id}'


def create_gemma_sae_cfg(layer_n = 9, width_k = 16):
    assert layer_n in [9, 20, 31]
    assert width_k in [16, 131]

    return SAEConfig(
        model_name = 'gemma-2-9b-it',
        sae_release = 'gemma-scope-9b-it-res-canonical',
        sae_id = f"layer_{layer_n}/width_{width_k}k/canonical",
        neuronpedia_id = f'gemma-2-9b-it/{layer_n}-gemmascope-res-{width_k}k',
        hook_name = f"blocks.{layer_n}.hook_resid_post",
        dataset_name = 'monology/pile-uncopyrighted',
        d_sae = 16384,
        context_size = 1024,
        normalize_activations = False
    )

def create_gpt2_sae_cfg(layer_n: int):
    return SAEConfig(
        model_name = 'gpt2-small',
        sae_release = 'gpt2-small-res-jb',
        sae_id = f"blocks.{layer_n}.hook_resid_pre",
        neuronpedia_id = f'gpt2-small/{layer_n}-res-jb',
        hook_name = f"blocks.{layer_n}.hook_resid_pre",
        dataset_name = 'Skylion007/openwebtext',
        d_sae = 24576,
        context_size = 128,
        normalize_activations = False
    )

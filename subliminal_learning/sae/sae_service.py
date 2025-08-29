from typing import Any
import gc
import tqdm
import torch

from sae_lens import SAE
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer

from subliminal_learning.llm import LLMConfig, ChatMessage, to_prompt
from subliminal_learning.sae import SAEActivations, SAEConfig, get_device

EXTRA_SPECIAL_TOKENS = ['\n', '.']

'''
TRANFORMERLENS / SAELENS SAE SERVICE

Pulls activations and runs them through the associated SAE. Many methods are similar to LLMService but with extensions. 

PARAMETERS
llm_config: LLMConfig
    LLMService configuration for Transformer Lens loading

sae_configs: List[SAEConfig]
    List of SAE configs for SAELens loading
    

METHODS
prompt_with_activations; batch_prompt_with_activations
    Returns SAEActivations objects for single prompt or multiple; does not apply chat template

chat_with_activations, batch_chat_with_activations
    Wraps prompt_with_activations/batch_prompt_with_activations but applies chat template and accepts ChatMessage list


'''


def get_tl_id(model_name: str):
    '''Transformer lens will only accept these names; no LoRA kwargs accepted'''
    return {
        "google/gemma-2-2b-it": "google/gemma-2-2b-it",
        "google/gemma-2-9b-it": "google/gemma-2-9b-it",
        "gpt2-small": "gpt2"
    }.get(model_name, model_name)


def recon_loss(target_acts: torch.Tensor, recon_acts: torch.Tensor) -> torch.Tensor:

    assert target_acts.shape == recon_acts.shape, f"Activations are different sizes! {recon_acts.shape} {target_acts.shape}"
    assert len(target_acts.shape) == 3, f"Activations must be 3 dimensional: {target_acts.shape} {recon_acts.shape}"

    x = torch.mean((target_acts - recon_acts)**2, dim = (-2, -1)) 

    return x.detach()


class SAEService:
    def __init__(
            self, 
            llm_config: LLMConfig, 
            sae_configs: list[SAEConfig], 
            extra_special_tokens: list[str] = EXTRA_SPECIAL_TOKENS
        ):

        self.name = 'saelens'

        self.llm_config = llm_config
        self.sae_configs = sae_configs
        self.extra_special_tokens = extra_special_tokens

        self.device = get_device()

        # Add SAE cache
        self.model = None
        self.tokenizer = None
        self.extra_special_tokens_ids = []
        self.saes = {}
    

    def _get_tokenizer(self, llm_config: LLMConfig) -> AutoTokenizer:
        '''Load the tokenizer for the model'''
        tl_model_id = get_tl_id(llm_config.base_model_id)
        return AutoTokenizer.from_pretrained(tl_model_id, device = self.device)


    def _get_model(self, llm_config: LLMConfig):

        assert len(llm_config.lora_kwargs) == 0, "LoRA not supported for HookedTransformer"
        assert len(llm_config.steering_vectors) == 0, "Steering not currently implemented"
        assert llm_config.base_model_id is not None

        tl_model_id = get_tl_id(self.llm_config.base_model_id)

        model = HookedTransformer.from_pretrained(
            tl_model_id, 
            device = self.device,
            dtype = 'float16',
            fold_ln = False,
            center_writing_weights = False,
            center_unembed = False
        )
        return model


    def get_tokenizer(self):
        self.tokenizer = self._get_tokenizer(self.llm_config)
        self.extra_special_tokens_ids = set([self.tokenizer.encode(x, add_special_tokens = False)[-1] for x in self.extra_special_tokens])
        print('Tokenizer loaded')


    def get_model_tokenizer(self):
        '''Load and store model and tokenizer'''

        if self.tokenizer is None:
            self.get_tokenizer()
        
        if self.model is None:
            self.model = self._get_model(self.llm_config)
            print('Model loaded')
        
        assert (self.model is not None) and (self.tokenizer is not None), "Experienced error loading model and tokenizer"


    def _get_sae(self, sae_config: SAEConfig) -> Any:
        return SAE.from_pretrained(
            release = sae_config.sae_release,
            sae_id = sae_config.sae_id,
            device = self.device,
        )


    def get_sae(self):
        '''Load SAEs if not already loaded'''
        
        for sae_config in self.sae_configs: 
            if sae_config.sae_name() not in self.saes:
                sae = self._get_sae(sae_config).to(self.device)
                sae.eval()
                self.saes[sae_config.sae_name()] = sae
                print('SAE loaded: ', sae_config.sae_name())
        
        assert len(self.saes) == len(self.sae_configs), "Experienced error loading saes, missing at least one SAE"


    def index_last_nonspecial_token(self, token_ids: torch.Tensor) -> int:
        '''Get the last token of the input tokens'''
        specials = set(self.tokenizer.all_special_ids or []).union(set(self.extra_special_tokens_ids))
        i = len(token_ids) - 1
        while i >= 0 and token_ids[i] in specials:
            i -= 1
        return i


    def prompt_with_activations(self, prompt: str, store_complete: bool = False, last_token_only: bool = False, debug: bool = False, **kwargs) -> SAEActivations:
        '''Retrieve activations for specific prompt

        :param: prompt: str: Prompt to cache activations for
        :param: store_complete: bool: Whether to save all activations or just SAE-encoded activations
        :param: debug: bool: Print debbugging statements

        Returns SAEActivations object with some arguments potentially empty depending on store_complete
        '''

        self.get_model_tokenizer()
        self.get_sae()

        # Get list of required hook names
        required_hooks = [sae_config.hook_name for sae_config in self.sae_configs]
        required_n_ids = [sae_config.neuronpedia_id for sae_config in self.sae_configs]
        if debug:
            print(f"Required hooks: {required_hooks}")

        # Use the tokenizer to convert it to tokens
        inputs = self.tokenizer.encode(prompt, return_tensors = "pt", add_special_tokens = True).to(self.device)
        last_token_idx = self.index_last_nonspecial_token(inputs[0])
        if debug:
            print("Inputs tokenized: ", str(inputs.shape),  str(inputs))
            print('Special tokens: ', self.extra_special_tokens_ids, self.tokenizer.all_special_tokens)
            print('Last token index: {last_token_idx} out of max index {len(inputs) - 1}')

        target_acts = []
        sae_acts = []
        recon_acts = []

        # Run model with selective caching
        with torch.no_grad():
            # Only cache the hooks we need
            _, cache = self.model.run_with_cache(
                inputs, 
                names_filter = lambda name: name in required_hooks
            )
            
            # Process each SAE config
            for sae_config in self.sae_configs:
                hook_name = sae_config.hook_name
                sae = self.saes[sae_config.sae_name()]

                # Get the target activations from cache
                target_act = cache[hook_name].clone()
                sae_act = sae.encode(target_act)

                
                if last_token_only:
                    target_act = target_act[:, last_token_idx].unsqueeze(1)
                    sae_act = sae_act[:, last_token_idx].unsqueeze(1)

                    if debug:
                        print(target_act.shape)
                        print(sae_act.shape)


                target_acts.append(target_act.cpu()) # Move to CPU to save GPU memory
                sae_acts.append(sae_act.cpu())
                recon_acts.append(sae.decode(sae_act).cpu())

                del target_act, sae_act

                if debug:
                    print(f'Captured activations for: {sae_config.sae_name()}')

            target_acts = torch.vstack(target_acts).cpu()
            recon_acts = torch.vstack(recon_acts).cpu()
                
            # Store results
            sae_activations = SAEActivations(
                input_str = prompt,
                input_tokens = inputs.cpu(),
                hook_names = required_hooks,
                neuronpedia_ids = required_n_ids, 
                last_token_idx = last_token_idx,
                last_token_only = last_token_only,
                target_acts = target_acts if store_complete else torch.Tensor([]),
                sae_acts = torch.vstack(sae_acts).cpu(),
                recon_acts = recon_acts if store_complete else torch.Tensor([]),
                recon_loss = recon_loss(
                    target_acts = target_acts,
                    recon_acts = recon_acts
                )
            )

        # Free up memory
        del cache, target_acts, sae_acts, recon_acts
        self.clear_cache()

        return sae_activations
    

    def batch_prompt_with_activations(self, prompts: list[str], **kwargs):

        if (self.model is None) or (self.tokenizer is None):
            self.get_model_tokenizer()
            print('Model and tokenizer loaded')
        assert (self.model is not None) and (self.tokenizer is not None), "Experienced error loading model and tokenizer"

        if self.saes is None:
            self.get_sae()
            print('SAEs loaded: ' + ', '.join(list(self.saes.keys())) )
        assert self.saes is not None, f"Error loading saes, no saes found for: {self.sae_configs}"


        sae_activations = []
        for prompt in tqdm.tqdm(prompts, desc='Retrieving activations'):
            sae_activations.append(self.prompt_with_activations(
                prompt = prompt,
                **kwargs
            ))

        return sae_activations


    def chat_with_activations(self, message_set: list[ChatMessage], store_complete = False, **kwargs):
        '''Chat + CACHE ACTIVATIONS - Memory optimized version'''

        self.get_model_tokenizer()
        assert (self.model is not None) and (self.tokenizer is not None), "Experienced error loading model and tokenizer"

        prompt = to_prompt(
            model_cfg = self.llm_config,
            messages = message_set,
            tokenizer = self.tokenizer,
            add_generation_prompt = False
        )

        return self.prompt_with_activations(
            prompt = prompt,
            store_complete = store_complete,
            **kwargs
        )


    def batch_chat_with_activations(self, messages: list[list[ChatMessage]], store_complete = False, **kwargs):

        self.get_model_tokenizer()
        self.get_sae()

        prompts = [
            to_prompt(
                model_cfg = self.llm_config,
                messages = message_set,
                tokenizer = self.tokenizer,
                add_generation_prompt = False
            ) for message_set in messages
        ]

        return self.batch_prompt_with_activations(
            prompts = prompts,
            store_complete = store_complete,
            **kwargs
        )
    

    def get_sae_features(self, activations: torch.Tensor, select_sae: str | list[str] | None = None):
        '''Returns SAE encoded features for a set of activations
        
        activations have dimension n_layers x n_tokens x activations_dim
        '''

        sae_acts = []
        recon_acts = []

        if select_sae is None:
            sae_list = list(self.sae_configs.keys())
        else:
            if isinstance(select_sae, str):
                sae_list = [select_sae]
            else:
                sae_list = select_sae

        # Process each SAE config
        for sae_config in self.sae_configs:
            if sae_config.sae_name() in sae_list:
                sae = self.saes[sae_config.sae_name()].to(self.device)

                # Get the target activations from cache
                sae_act = sae.encode(activations.to(self.device)) 
                recon_act = sae.decode(sae_act)

                sae_acts.append(sae_act.detach().unsqueeze(0))
                recon_acts.append(recon_act.detach().unsqueeze(0))

                del sae_act, recon_act

        sae_acts = torch.vstack(sae_acts)
        recon_acts = torch.vstack(recon_acts)

        return sae_acts, recon_acts # Shapes: num_saes x (activations dimensions)



    def clear_cache(self):
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.mps.is_available():
            torch.mps.empty_cache()
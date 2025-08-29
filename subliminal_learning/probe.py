import torch
import torch.nn as nn
import torch.optim as optim

import traceback
import polars as pl
import numpy as np
from pydantic import BaseModel

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, f1_score, recall_score

from subliminal_learning.sae import get_device

device = get_device()


'''
LINEAR PROBE

Creates a basic SLP on activations for classification. Allows for binary or multiclass
Includes analysis functions for metrics calculation

'''


class ProbeConfig(BaseModel):
    layer: int
    class_names: list[str] # Control class must be listed first if using use_multiclass = True
    activations_dim: int # Hidden dimension for given model

    seed: int = 42
    use_multiclass: bool = False # Set to False to use binary classificaiton

    test_size: float = 0.20
    validation_size: float = 0.20

    batch_size: int = 32

    n_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    device: str = device
    debug: bool = False # Adds prints statements to data loader

    @property
    def class_to_name_map(self) -> dict:
        if self.use_multiclass or (len(self.class_names) == 2):
            return dict(zip(range(len(self.class_names)), self.class_names))
        else:
            return {0: 'control', 1: 'animal'} # Multi-class assumes control class was first
        
    @property
    def name_to_class_map(self) -> dict:
        if self.use_multiclass or (len(self.class_names) == 2):
            return dict(zip(self.class_names, range(len(self.class_names))))
        else:
            return {k: 1 if k != 'control' else 0 for k in self.class_names} # Multi-class assumes control class was first

    
    @property
    def n_classes(self) -> int:
        if self.use_multiclass:
            return len(self.class_names)
        else:
            return 2


# Prepare the data for binary classification (otter vs dog)
def prepare_data(raw_activations: pl.DataFrame, probe_cfg: ProbeConfig):
    ''' Prepare activations data for classification
    
    raw_activations: Output of SAE post-processing activations script; polars dataframe with columns target and act_0, act_1, .... act_N where N is the hidden dimension
    probe_cfg: ProbeConfig: Configuration 
        
    returns dictionary with keys test, validation, eval - each containing a dataloader
    '''
    # If layer = -1 then use all layers
    if probe_cfg.layer == -1:
        # Concat all of the layer activations
        activations = raw_activations.filter(
            (pl.col('target').is_in(probe_cfg.class_names))
        ).pivot(
            index = ['id', 'target'],
            on = 'layer',
            values = [x for x in raw_activations.columns if str(x).startswith('act_')]
        )

        # Modify the config activations dim so that the model is the correct size
        probe_cfg.activations_dim = len([x for x in activations.columns if str(x).startswith('act_')])
    else:
        activations = raw_activations.filter(
            (pl.col('layer') == probe_cfg.layer)
            & (pl.col('target').is_in(probe_cfg.class_names))
        )

    activations = activations.with_columns(
        pl.col('target').replace_strict(probe_cfg.name_to_class_map).cast(pl.Int32).alias('label')
    )
    class_sizes = activations.group_by('label').agg(pl.len().alias('count')).to_pandas().set_index('label')['count'].to_dict()
    
    if probe_cfg.debug:
        print(f"Class sizes: {class_sizes}")
    
    # Extract activation columns
    act_columns = [x for x in activations.columns if str(x).startswith('act_')]
    
    # Create features and labels
    X = torch.Tensor(activations.select(*act_columns).to_numpy())
    y = torch.Tensor(activations['label'].to_numpy())
    
    if probe_cfg.debug:
        print(f"Input dim: {X.shape}")
        print(f"Target dimension: {y.shape}")
    
    # Split test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size = probe_cfg.test_size, 
        random_state = probe_cfg.seed, 
        stratify = y,
    )
    
    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, 
        test_size = probe_cfg.validation_size / (1 - probe_cfg.test_size), 
        random_state = probe_cfg.seed, 
        stratify = y_temp,
    )
    
    if probe_cfg.debug:
        print(f"Train samples: {X_train.shape[0]}")
        print(f"Validation samples: {X_val.shape[0]}")
        print(f"Test samples: {X_test.shape[0]}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size = probe_cfg.batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = probe_cfg.batch_size, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = probe_cfg.batch_size, shuffle = False)
    
    return {
        'train': train_loader,
        'validation': val_loader,
        'test': test_loader
    }



class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes = 2):
        ''' Single layer MLP for classification from SAE activations
        
        input_dim: Dimension of input features (default 3584 for Gemma 2 9B activations)
        num_classes: Number of output classes (default 2 for binary classification)
        
        '''
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Single linear layer with dropout
        self.classifier = nn.Linear(input_dim, num_classes)

        
    def forward(self, x):
        return self.classifier(x)
    

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.softmax(logits, dim = -1)
            return probabilities
    
    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim = -1)
            return predictions
        


def train(model: LinearProbe, training_data: DataLoader, validation_data: DataLoader, probe_cfg: ProbeConfig, in_notebook: bool = True, use_tqdm: bool = False):
    ''' Train the linear probe
    
    model: LinearProbe instance
    training_data: DataLoader for training data
    validation_data: DataLoader for validation data
    probe_cfg: ProbeConfig containing training specs
    in_notebook: bool: Allows tqdm to work even when in notebook


    returns dictionary of loss values recorded during ttraining
    '''

    if in_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    device = probe_cfg.device

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr = probe_cfg.learning_rate, 
        weight_decay = probe_cfg.weight_decay
    )
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    if use_tqdm:
        pbar = tqdm(range(probe_cfg.n_epochs), desc = 'Running Probe Training')
    else:
        pbar = range(probe_cfg.n_epochs)

    for epoch in pbar:
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        
        for batch_x, batch_y in training_data:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in validation_data:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim = -1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        epoch_train_loss /= len(training_data)
        epoch_val_loss /= len(validation_data)
        val_acc = accuracy_score(all_labels, all_preds)
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(val_acc)
        
        if use_tqdm:
            pbar.set_postfix_str(f'Epoch [{epoch+1}] - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {val_acc:.4f}')
    

    return {
        'training_loss': train_losses,
        'validation_loss': val_losses,
        'validation_accuracy': val_accuracies
    }



def evaluate(model: LinearProbe, test_data: DataLoader, probe_cfg: ProbeConfig, in_notebook: bool = True, use_tqdm: bool = True):
    '''Evaluate the linear probe
    
    model: Trained LinearProbe instance
    test_data: DataLoader for test data
    probe_cfg: Configuration of the probe
    in_notebook: bool: Used to allow tqdm to work in notebook
    
    Returns dictionary:
        predictions: Predicted labels of test set
        labels: True labels of test set
        probabilities: Probabilities from predictions
        accuracy: Accuracy score
        roc_auc: ROC AUC score
    '''
    if in_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    device = probe_cfg.device
    
    model.eval()
    pred_labels = []
    true_labels = []
    all_probs = []
    
    with torch.no_grad():

        if use_tqdm:
            pbar = tqdm(test_data, desc = 'Running Probe Evaluation')
        else:
            pbar = test_data

        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim = -1)
            preds = torch.argmax(outputs, dim = -1)
            
            pred_labels.extend(preds.cpu().numpy())
            true_labels.extend(batch_y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_probs = np.array(all_probs)
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)

    return {
        'pred_labels': pred_labels,
        'true_labels': true_labels,
        'probs': all_probs,

    }


def calculate_metrics(test_results: dict, by_class = False):

    true_labels, pred_labels, probs = test_results['true_labels'], test_results['pred_labels'], test_results['probs']

    if probs.shape[1] == 2:
        probs = probs[:, 1]
        multi_class = 'raise'
    else:
        multi_class = 'ovr'

    if by_class:
        average = None
    else:
        average = 'macro' # rebalances labels (unweighted mean across labels)

    metrics = {
        'accuracy': accuracy_score(true_labels, pred_labels), # if you separate by label, either you are getting precision or getting recall
        'roc_auc':  roc_auc_score(true_labels, probs, multi_class = multi_class, average = average),
        'precision': precision_score(true_labels, pred_labels, average = average), # True positive / all positive
        'recall': recall_score(true_labels, pred_labels, average = average), # True positive / all actual label
        'f1_score': f1_score(true_labels, pred_labels, average = average),
    }

    if by_class:
        return {
            k: {m: metrics[m][k] if not isinstance(metrics[m], float) else metrics[m] for m in metrics } for k in range(len(metrics['f1_score']))
        }
    else:
        return metrics



def run_probe(raw_activations: pl.DataFrame, probe_cfg: ProbeConfig, in_notebook: bool = True, use_tqdm: bool = False, raise_exceptions: bool = False):
    try:
        # Prepare the data
        data_loaders = prepare_data(
            raw_activations, 
            probe_cfg = probe_cfg
        )

        # Create the model
        model = LinearProbe(
            input_dim = probe_cfg.activations_dim, 
            num_classes = probe_cfg.n_classes
        )

        # Run Training
        training_history = train(
            model = model,
            training_data = data_loaders['train'],
            validation_data = data_loaders['validation'],
            probe_cfg = probe_cfg,
            in_notebook = in_notebook,
            use_tqdm = use_tqdm
        )

        # Evaluate on the test set
        test_results = evaluate(
            model = model,
            test_data = data_loaders['test'],
            probe_cfg = probe_cfg,
            in_notebook = in_notebook,
            use_tqdm = use_tqdm
        )


        return {
            'training_history': training_history,
            'test_results': test_results
        }
    except KeyboardInterrupt as e:
        raise e
    except BaseException as e:
        if raise_exceptions:
            raise e
        else:
            print(traceback.format_exc())
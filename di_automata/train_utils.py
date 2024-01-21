"""Training, eval, dataloader setup and model setup."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split

from ib_fcnn.architectures.mlp import *
from ib_fcnn.config_setup import *
from ib_fcnn.datasets.parity import ParityDataset
from ib_fcnn.datasets.palindrome import PalindromeDataset

from typing import Tuple, List, Union

# Path to root dir (with setup.py)
PROJECT_ROOT = Path(__file__).parent.parent

def check_valid_dataloader(dataloader: DataLoader) -> None:
    try:
        iter(dataloader)
    except TypeError:
        raise ValueError("Invalid dataloader provided")
    

def save_model(
    path: str, 
    epoch: int, 
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    train_loss: List[float], 
    train_acc: List[float], 
    test_acc: List[float],
    final_acc: float,
    test_preds: str
    ) -> None:
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_hist': train_loss,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'final_acc': final_acc,
                'test_preds': test_preds
                },
                path)
    

def train(model: nn.Module, 
          train_loader: DataLoader, 
          test_loader: DataLoader, 
          default_lr: float, 
          epochs: int, 
          loss_threshold: float,
          num_eval_batches: int,
          optimizer: optim.Optimizer,
          project: str,
          model_save_path: str,
          device: torch.device = torch.device('cuda')
          ) -> None:
    """
    Assume model already on device and optimizer already initialised with model parameters, lr and momentum (see optimizer config).
    Args:
        lr: Fixed based LR. See cosine LR in optimizer for more information on types of LR.
        epochs: Max epochs to train for. In this configuration where training is halted after a certain loss is achieved, often training will not reach this upper bound.
        loss_threshold: Training will stop when the batch average training loss is below this threshold.
        optimizer: Optimizer to use for training.
        project: Name of the project for wandb.
        model_save_path: Path to save the model.
    """
    criterion = nn.BCEWithLogitsLoss()
    test_acc_list, train_acc_list = [], []

    for epoch in range(epochs):
        print("Epoch: ", epoch)
        train_loss = []
        model.train()
        
        for inputs, labels in tqdm(train_loader, desc=f"Training iterations within epoch {epoch}"):
            inputs, labels = inputs.to(device), labels.to(device)
            scores = model(inputs)

            optimizer.zero_grad()
            loss = criterion(scores.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # Evaluate at end of epoch
        train_acc = evaluate(model, train_loader, subset=True, num_eval_batches=num_eval_batches)
        test_acc, fn_rep = evaluate(model, test_loader, subset=False)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Project {project}, Epoch: {epoch}, Train accuracy: {train_acc}, Test accuracy: {test_acc}, LR {default_lr}, Loss Threshold: {loss_threshold}')
        wandb.log({"Train Acc": train_acc, "Test Acc": test_acc, "Loss": np.mean(train_loss), "LR": default_lr, "Function Representation on Test Set": fn_rep}, step=epoch)
        
        avg_train_loss = np.mean(train_loss)
        if avg_train_loss < loss_threshold:  # check if the average training loss is below the threshold
            print(f'Average training loss {avg_train_loss:.3f} is below the threshold {loss_threshold}. Training stopped.')
            break

    final_acc, test_preds = evaluate(model, test_loader)
    save_model(f"{model_save_path}_final", epoch, model, optimizer, train_loss, train_acc_list, test_acc_list, final_acc, test_preds)
    

@torch.no_grad()
def evaluate(
    model: nn.Module, 
    data_loader: DataLoader,
    device: torch.device = torch.device("cuda"),
    subset: bool = False, 
    num_eval_batches: int = None) -> Union[float, Tuple[float, str]]:
    """"
    Args:
        subset: Whether to evaluate on whole dataloader or just a subset.
        num_eval_batches: If we aren't evaluating on the whole dataloader, then do on this many batches.
    Returns:
        accuracy: Percentage accuracy.
        test_preds: List of predictions of model on test set as a binary string. For Boolean functions only.
    """
    if subset and num_eval_batches is None:
        raise ValueError("For subset evaluation, num_eval_batches should be specified.")

    model = model.to(device).eval()
    predictions = []
    actuals = []

    for batch_index, (inputs, labels) in enumerate(data_loader):
        if subset and batch_index >= num_eval_batches:  # Limit the number of batches processed by the function
            break
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        output = torch.sigmoid(output).squeeze()
        predicted = (output > 0.5).cpu()  # Threshold sigmoid outputs for binary predictions
        predictions.append(predicted)
        actuals.append(labels.cpu())

    model.train()

    # Convert lists of tensors to single tensor
    predictions, actuals = torch.cat(predictions), torch.cat(actuals)
    accuracy = (predictions == actuals).float().mean().item() * 100

    if subset: # For train accuracy
        return accuracy
    else: # For evaluating on test set only
        test_preds = ''.join([str(int(p)) for p in predictions.tolist()])
        return accuracy, test_preds


# TODO
def calculate_gp():
    pass


def create_or_load_dataset(dataset_type: str, dataset_config: DatasetConfig) -> Dataset:
    """Create or load an existing dataset based on a specified filepath and dataset type."""
    filepath = f'{dataset_config.data_folder}/{dataset_config.filename}.pt'
    if os.path.exists(filepath):
        dataset = torch.load(filepath)
    else:
        dataset_type = globals()[dataset_type]
        dataset = dataset_type(dataset_config)
        torch.save(dataset, filepath)
    return dataset


def create_dataloaders(
    dataset: Dataset,
    dl_config: DataLoaderConfig) -> Tuple[DataLoader, DataLoader]:
    """For a given dataset and dataloader configuration, return test and train dataloaders with a deterministic test-train split on full dataset (set by seed - see configs)."""
    assert 0 <= dl_config.train_fraction <= 1, "train_fraction must be between 0 and 1."
    torch.manual_seed(dl_config.seed)
    train_size = int(dl_config.train_fraction * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=dl_config.train_bs, shuffle=dl_config.shuffle_train)
    test_dataloader = DataLoader(test_dataset, batch_size=dl_config.test_bs)   
    return train_dataloader, test_dataloader


def save_to_csv(
    dataset: Dataset, 
    filename: str,
    input_col_name: str ='input',
    label_col_name: str ='label') -> None:
    data = [(str(x.numpy()), int(y.numpy())) for x, y in dataset]
    df = pd.DataFrame(data, columns=[input_col_name, label_col_name])
    df.to_csv(filename, index=False)
    

# TODO: complete pass
def model_constructor(config: MainConfig) -> nn.Module:
    """Constructs a model based on a specified model type."""
    if config.model_type == ModelType.MLP:
        model = mlp_constructor(
            input_size=config.dataset.input_length,
            hidden_sizes=config.mlp_config.hidden_sizes,
            output_size=config.mlp_config.output_size,
            bias=config.mlp_config.add_bias,
        )
    elif config.model_type == ModelType.RNN:
        pass
    elif config.model_type == ModelType.LSTM:
        pass
    else:
        raise ValueError(f"Invalid model type: {config.model_type}")
    return model


def optimizer_constructor(config: MainConfig, model: nn.Module) -> optim.Optimizer:
    match config.optimization.optimizer_type:
        case OptimizerType.SGD:
            optim_constructor = torch.optim.SGD
        case OptimizerType.ADAM:
            optim_constructor = torch.optim.Adam
        case _:
            raise ValueError(f"Unknown optimizer type: {config.optimization.optimizer_type}")
    optim = optim_constructor(
        params=model.parameters(),
        lr=config.optimization.default_lr,
        **config.optimization.optimizer_kwargs,
    )
    return optim


def get_input_shape(dataset: Dataset) -> tuple[int, ...]:
    """
    Get the input shape of a dataset. This is useful for constructing the model.

    Assumes a supervised dataset (x, y)
    """
    return dataset[0][0].shape
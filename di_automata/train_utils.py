""""Make Run class which contains all relevant logic for instantiating and training."""
from typing import Tuple, List, Dict, TypedDict, TypeVar
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import logging
import os
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from di_automata.plotting.plot_utils import visualise_seq_data
from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.optim.sgnht import SGNHT
from di_automata.devinterp.slt import estimate_learning_coeff
from di_automata.tasks.data_utils import take_n
from di_automata.config_setup import *
from di_automata.constructors import (
    construct_model, 
    optimizer_constructor, 
    initialise_model,
    create_dataloader_hf
)
Sweep = TypeVar("Sweep")

# Path to root dir (with setup.py)
PROJECT_ROOT = Path(__file__).parent.parent


class StateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Dict
    
    
def state_dict(model, optimizer, scheduler) -> StateDict:
    """Used for model saving.
    
    Note if cosine LR scheduler not used, scheduler is None.
    """
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": {k: v for k,v in scheduler.state_dict().items() if not callable(v)} if scheduler is not None else None,
    }
    

class Run:
    def __init__(
        self, 
        config: MainConfig,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.config = config
        
        self._set_logger()
        
        self.train_loader = create_dataloader_hf(self.config)
        
        self.model, param_inf_properties = construct_model(config)
        self.model.to(self.device)
        # Initialise the model (possibly with muP, link: https://arxiv.org/pdf/2203.03466.pdf)
        initialise_model(config, self.model, param_inf_properties)
        
        self.optimizer, self.scheduler = optimizer_constructor(config, self.model, param_inf_properties)
        
        self.model_save_dir = Path(self.config.model_save_path)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
    
    def train(self) -> None:
        criterion = nn.CrossEntropyLoss()
        self.train_acc_list, self.train_loss_list = [], []
        self.model.train()
        self.progress_bar = tqdm(total=self.config.num_training_iter)
        self.idx, self.epoch = 0, 0
        
        for epoch in range(self.config.num_epochs):
            self.epoch += 1
            train_loss = []
            print(f"Training epoch {epoch}")
            num_iter = self.config.eval_frequency if epoch + 1 <= self.config.num_training_iter / self.config.eval_frequency  else self.config.num_training_iter % self.config.eval_frequency
            
            for data in take_n(self.train_loader, num_iter): # Compatible with HF dataset format where data is a dictionary
                # all_idxs is a list of idxs (not useful)
                self.idx += 1
                inputs, labels = data["input_ids"].to(self.device), data["label_ids"].to(self.device)
                logits = self.model(inputs)

                self.optimizer.zero_grad()
                loss = criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.lr = self.scheduler.get_last_lr()[0]
                train_loss.append(loss.item())
                self.progress_bar.update()
                
            train_loss = self._evaluation_step(epoch)
            
            avg_train_loss = np.mean(train_loss)
            if avg_train_loss < self.config.loss_threshold:
                print(f'Average training loss {avg_train_loss:.3f} is below the threshold {self.config.loss_threshold}. Training stopped.')
                break

            self._save_model()
        
        if self.config.is_wandb_enabled:
            wandb.finish()
    
    def _evaluation_step(self, epoch) -> float:
        """TODO: consider test accuracy and loss."""
        train_acc, train_loss = evaluate(
            model=self.model, 
            data_loader=self.train_loader, 
            num_eval_batches=self.config.num_eval_batches,
            idx=self.idx
        )
        self.train_acc_list.append(train_acc)
        self.train_loss_list.append(train_loss)
        # test_acc, test_loss = evaluate(self.model, self.test_loader, subset=False)
        # self.test_acc_list.append(test_acc)
        
        self.progress_bar.set_description(f'Project {self.config.wandb_config.wandb_project_name}, Epoch: {epoch}, Train Accuracy: {train_acc}, Train Loss: {train_loss}, LR {self.lr}, Loss Threshold: {self.config.loss_threshold}')
        
        # wandb.log({"Train Acc": train_acc, "Test Acc": test_acc, "Train Loss": np.mean(train_loss), "Test Loss": np.mean(test_loss), "LR": self.config.optimizer_config.default_lr}, step=epoch)
        wandb.log({"Train Acc": train_acc, "Train Loss": train_loss, "LR": self.config.optimizer_config.default_lr}, step=epoch)
        
        return train_loss

    # TODO: edit config run name in config setup file
    def _save_model(self):
        if self.config.wandb_config.save_model_as_artifact:
            model_artifact = wandb.Artifact(f"epoch_{self.epoch}", type="model", description="The trained model state_dict")
            model_artifact.add_file(".hydra/config.yaml", name="config.yaml")
            wandb.log_artifact(model_artifact)
        if self.config.save_local:
            file_path =  self.model_save_dir / f"{self.config.run_name}"
            state = state_dict(self.model, self.optimizer, self.scheduler)
            data = {"Train Acc": self.train_acc_list, "Train Loss": self.train_loss_list}
            torch.save(dict(state, **data), file_path)
            
    def _set_logger(self) -> None:
        """Currently uses wandb as default."""
        logging.info(f"Hydra current working directory: {os.getcwd()}")
        logger_params = {
            "name": self.config.run_name,
            "project": self.config.wandb_config.wandb_project_name,
            "settings": wandb.Settings(start_method="thread"),
            "config": OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
            "mode": "disabled" if not self.config.is_wandb_enabled else "online",
        }
        wandb.init(**logger_params, entity="wu-cindyx")
        # Probably won't do sweeps over these - okay to put here relative to call to update_with_wandb_config() below
        wandb.config.dataset_type = self.config.dataset_type
        wandb.config.model_type = self.config.model_type
    
    def _clean_sweep(self, sweep: "Sweep") -> List["Sweep"]:
        """Get rid of wandb runs that crashed with _step = None."""
        def _clean_sweep():
            for r in sweep.runs:
                if r.summary.get("_step", None) is None:
                    r.delete()
                    yield r

        return list(r for r in _clean_sweep())
    
    def restore(self) -> None:
        """TODO: implement restoring from last checkpoint."""
        pass
    

@torch.no_grad()
def evaluate(
    model: nn.Module, 
    data_loader: DataLoader,
    num_eval_batches: int,
    idx: int,
    device: torch.device = torch.device("cuda"),
) -> Tuple[float, float]:
    """"
    Args:
        subset: Whether to evaluate on whole dataloader or just a subset.
        num_eval_batches: If we aren't evaluating on the whole dataloader, then do on this many batches.
    Returns:
        accuracy: Average percentage accuracy on batch.
        loss: Average loss on batch.
    """
    model = model.to(device).eval()
    total_accuracy = 0.
    total_loss = 0.
    assert isinstance(idx, int), "idx must be an int"

    for data in take_n(data_loader, num_eval_batches):
        inputs, labels = data["input_ids"].to(device), data["label_ids"].to(device)
        visualise_seq_data(inputs, idx)
        outputs = model(inputs)
        
        total_loss += F.cross_entropy(outputs, labels)
        # Second dimension is class dimension in PyTorch for sequence data (see AutoGPT transformer for details)
        probs = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probs, dim=1)

        correct_predictions = predictions == labels
        total_accuracy += correct_predictions.float().mean().item() * 100

    model.train()
    return total_accuracy / num_eval_batches, (total_loss / num_eval_batches).item()


def update_with_wandb_config(config: OmegaConf, sweep_params: list[str]) -> OmegaConf:
    """Check if each parameter exists in wandb.config and update it if it does."""
    for param in sweep_params:
        if param in wandb.config:
            print("Updating param with value from wandb config: ", param)
            OmegaConf.update(config, param, wandb.config[param], merge=True)
    return config


# #TODO: need?
# def create_dataloaders(
#     dataset: Dataset,
#     dl_config: DataLoaderConfig,
# ) -> Tuple[DataLoader, DataLoader]:
#     """For a given dataset and dataloader configuration, return test and train dataloaders with a deterministic test-train split on full dataset (set by seed - see configs)."""
#     assert 0 <= dl_config.train_fraction <= 1, "train_fraction must be between 0 and 1."
#     torch.manual_seed(dl_config.seed)
#     train_size = int(dl_config.train_fraction * len(dataset))
#     test_size = len(dataset) - train_size
#     train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
#     train_dataloader = DataLoader(train_dataset, batch_size=dl_config.train_bs, shuffle=dl_config.shuffle_train)
#     test_dataloader = DataLoader(test_dataset, batch_size=dl_config.test_bs)   
#     return train_dataloader, test_dataloader


# TODO: need?
def save_to_csv(
    dataset: Dataset, 
    filename: str,
    input_col_name: str ='input',
    label_col_name: str ='label'
) -> None:
    """Save a dataset to a csv file."""
    data = [(str(x.numpy()), int(y.numpy())) for x, y in dataset]
    df = pd.DataFrame(data, columns=[input_col_name, label_col_name])
    df.to_csv(filename, index=False)


def get_previous_commit_hash():
    """Used to get Github commit hash."""
    try:
        # Execute the git command to get the previous commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD~1'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output.decode('utf-8')}")
        return None
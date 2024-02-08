""""Make Run class which contains all relevant logic for instantiating and training."""
from typing import Tuple, List, Dict, TypedDict, TypeVar
import typing
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
import logging
import os
from einops import rearrange, reduce, repeat
import subprocess
from omegaconf import OmegaConf
from functools import partial
from sklearn.decomposition import PCA
import pandas as pd
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from di_automata.plotting.plot_utils import visualise_seq_data
from di_automata.devinterp.optim.sgld import SGLD
from di_automata.devinterp.optim.sgnht import SGNHT
from di_automata.devinterp.slt import estimate_learning_coeff, estimate_learning_coeff_with_summary
from di_automata.devinterp.rlct_utils import plot_components
from di_automata.tasks.data_utils import take_n
from di_automata.config_setup import *
from di_automata.constructors import (
    construct_model, 
    optimizer_constructor, 
    initialise_model,
    create_dataloader_hf,
    construct_rlct_criterion,
)
Sweep = TypeVar("Sweep")

# Path to root dir (with setup.py)
PROJECT_ROOT = Path(__file__).parent.parent


class StateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Dict
    
    
def get_state_dict(model, optimizer, scheduler) -> StateDict:
    """If cosine LR scheduler not used, scheduler is None."""
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
        self.config = config
        
        self.train_loader = create_dataloader_hf(self.config, deterministic=False)
        self.ed_loader = create_dataloader_hf(self.config, deterministic=True) # Make sure deterministic to see same data
        
        self.model, param_inf_properties = construct_model(config)
        self.model.to(self.device)
        # Initialise the model (possibly with muP, link: https://arxiv.org/pdf/2203.03466.pdf)
        initialise_model(config, self.model, param_inf_properties)
        # self.model = torch.compile(self.model) # Not supported for WIndows. Bill Gates computer.
        
        self.optimizer, self.scheduler = optimizer_constructor(config, self.model, param_inf_properties)
        
        self.model_save_dir = Path(self.config.model_save_path)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.rlct_sgld, self.rlct_sgnht, self.rlct_sgld_std, self.rlct_sgnht_std = {}, {}, {}, {} # dict[str, float]
        rlct_folder = Path(__file__).parent / self.config.rlct_config.rlct_data_dir
        rlct_folder.mkdir(parents=True, exist_ok=True)
        self.rlct_criterion = construct_rlct_criterion(self.config)
        
        self.ed_logits = []
        
        self._set_logger()
    
    def train(self) -> None:
        criterion = nn.CrossEntropyLoss()
        self.train_acc_list, self.train_loss_list = [], []
        self.model.train()
        self.progress_bar = tqdm(total=self.config.num_training_iter)
        self.idx, self.epoch = 0, 0
        self.lr = self.config.optimizer_config.default_lr
        
        for epoch in range(self.config.num_epochs):
            self.epoch += 1
            train_loss = []
            print(f"Training epoch {epoch}")
            self.num_iter = self.config.eval_frequency if epoch + 1 <= self.config.num_training_iter / self.config.eval_frequency  else self.config.num_training_iter % self.config.eval_frequency
            
            for data in take_n(self.train_loader, self.num_iter): # Compatible with HF dataset format where data is a dictionary
                # all_idxs is a list of idxs (not useful)
                self.idx += 1
                inputs, labels = data["input_ids"].to(self.device), data["label_ids"].to(self.device)
                logits = self.model(inputs)

                self.optimizer.zero_grad()
                loss = criterion(logits, labels.long())
                loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                    self.lr = self.scheduler.get_last_lr()[0]
                train_loss.append(loss.item())
                self.progress_bar.update()
                
            train_loss = self._evaluation_step()

            self._ed_data_training()
            
            avg_train_loss = np.mean(train_loss)
            if avg_train_loss < self.config.loss_threshold:
                print(f'Average training loss {avg_train_loss:.3f} is below the threshold {self.config.loss_threshold}. Training stopped.')
                break

            self._save_model()
    
    def _ed_data_training(self) -> None:
        """Collect essential dynamics logit data each epoch."""
        logits_epoch = []
        with torch.no_grad():
            for data in take_n(self.ed_loader, self.config.rlct_config.ed_config.batches_per_checkpoint):
                inputs = data["input_ids"].to(self.device)
                logits = self.model(inputs)
                # Flatten over batch, class and sequence dimension
                logits_epoch.append(rearrange(logits, 'b c s -> (b c s)'))
        self.model.train()
        
        # Concat all per-batch logits over batch dimension to form one super-batch
        logits_epoch = torch.cat(logits_epoch)
        self.ed_logits.append(logits_epoch)
        
        # Save and log to WandB
        torch.save(logits_epoch, "logits")
        logit_artifact = wandb.Artifact(f"logits", type="logits", description="The trained model state_dict")
        logit_artifact.add_file("logits", name="config.yaml")
        wandb.log_artifact(logit_artifact, aliases=[f"epoch{self.epoch}_{self.config.run_name}"])
        
    def ed_calculation(self):
        """PCA and plot part of ED."""
        pca = PCA(n_components=3)
        concat_logit_matrix = torch.stack(self.ed_logits)
        pca.fit(concat_logit_matrix.cpu().numpy())
        
        projected_1, projected_2, projected_3 = [], [], []
        for epoch in range(self.config.num_epochs):
            logits_epoch = rearrange(self.ed_logits[epoch], 'n -> 1 n').cpu().numpy()
            projected_vector = pca.transform(logits_epoch)[0]
            projected_1.append(projected_vector[0])
            projected_2.append(projected_vector[1])
            projected_3.append(projected_vector[2])

        p = plot_components(projected_1, projected_2, projected_3)
        p.save("plot.png", width=10, height=4, dpi=300)
        wandb.log({"ED_PCA": wandb.Image("plot.png")})
    
    def restore_model(self, resume_training: bool = False) -> None:
        """Restore from a checkpoint on wandb. 
        
        TODO: implement local storing?
        """
        artifact = self.run.use_artifact(f"{self.config.wandb_config.entity_name}/{self.config.wandb_config.wandb_project_name}/states:epoch{self.epoch}_{self.config.run_name}")
        data_dir = artifact.download()
        config_path = Path(data_dir) / "config.yaml"
        model_state_path = Path(data_dir) / "state.torch"
        config = OmegaConf.load(config_path)
        self.config = typing.cast(MainConfig, config)
        states = torch.load(model_state_path)
        
        self.model_state_dict, optimizer_state_dict, scheduler_state_dict = states["model"], states["optimizer"], states["scheduler"]
        
        if resume_training:
            # Only need below code if a) passing ref model in and doing deepcopy;
            # b) using this function outside of RLCT estimation for e.g. resuming training
            model, _ = construct_model(self.config)
            model.load_state_dict(self.model_state_dict)
            model.to(self.device)
            model.train()
            self.model = torch.compile(model)
            self.optimizer.load_state_dict(optimizer_state_dict)
            self.scheduler.load_state_dict(scheduler_state_dict)
            
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
        self.run = wandb.init(**logger_params, entity="wu-cindyx")
        # Probably won't do sweeps over these - okay to put here relative to call to update_with_wandb_config() below
        wandb.config.dataset_type = self.config.dataset_type
        wandb.config.model_type = self.config.model_type
    
    def _evaluation_step(self) -> float:
        """TODO: consider test accuracy and loss."""
        train_acc, train_loss = evaluate(
            model=self.model, 
            data_loader=self.train_loader, 
            num_eval_batches=self.config.num_eval_batches,
            idx=self.idx
        )
        self.train_acc_list.append(train_acc)
        self.train_loss_list.append(train_loss)
        
        self.progress_bar.set_description(f'Project {self.config.wandb_config.wandb_project_name}, Epoch: {self.epoch}, Train Accuracy: {train_acc}, Train Loss: {train_loss}, LR {self.lr}, Loss Threshold: {self.config.loss_threshold}')
        
        wandb.log({"Train Acc": train_acc, "Train Loss": train_loss, "LR": self.config.optimizer_config.default_lr}, step=self.idx)
        
        return train_loss

    def _save_model(self) -> None:
        model_to_save = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        state = get_state_dict(model_to_save, self.optimizer, self.scheduler)
        torch.save(
            state, # Save optimizer, model and scheduler all in one go
            Path(".") / "states.torch", # Working directory configured by Hydra as output directory
        )
        
        if self.config.wandb_config.save_model_as_artifact is True:
            model_artifact = wandb.Artifact(f"states", type="states", description="The trained model state_dict")
            model_artifact.add_file(f"states.torch")
            model_artifact.add_file(".hydra/config.yaml", name="config.yaml")
            wandb.log_artifact(model_artifact, aliases=[f"epoch{self.epoch}_{self.config.run_name}"])
        else:
            file_path =  self.model_save_dir / f"{self.config.run_name}"
            data = {"Train Acc": self.train_acc_list, "Train Loss": self.train_loss_list}
            torch.save(dict(state, **data), f"{file_path}.torch")
            
    def _clean_sweep(self, sweep: "Sweep") -> List["Sweep"]:
        """Get rid of wandb runs that crashed with _step = None."""
        def _clean_sweep():
            for r in sweep.runs:
                if r.summary.get("_step", None) is None:
                    r.delete()
                    yield r

        return list(r for r in _clean_sweep())
    
    def finish_run(self):
        if self.config.is_wandb_enabled:
            wandb.finish()
    
        
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
        
        total_loss += F.cross_entropy(outputs, labels.long())
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
        
        
def get_previous_commit_hash():
    """Used to get Github commit hash."""
    try:
        # Execute the git command to get the previous commit hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD~1'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        return commit_hash
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.output.decode('utf-8')}")
        return None
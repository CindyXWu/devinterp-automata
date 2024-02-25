import wandb
from typing import TypeVar
from pathlib import Path
from functools import partial
import shutil
import time
import pickle
import os
import s3fs
from einops import rearrange
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
from omegaconf import OmegaConf

import torch

from di_automata.devinterp.slt.sampler import estimate_learning_coeff_with_summary
from di_automata.devinterp.rlct_utils import (
    extract_and_save_rlct_data,
    plot_pca_plotly,
    plot_explained_var,
)
from di_automata.config_setup import *
from di_automata.constructors import (
    construct_model, 
    create_dataloader_hf,
    construct_rlct_criterion,
)
from di_automata.tasks.data_utils import take_n
from di_automata.io import read_tensors_from_file, append_tensor_to_file
from di_automata.devinterp.ed_utils import EssentialDynamicsPlotter, FormPotentialPlotter
Sweep = TypeVar("Sweep")

# AWS
s3 = s3fs.S3FileSystem()


class PostRunSLT:
    def __init__(self, slt_config: PostRunSLTConfig):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.slt_config: PostRunSLTConfig = slt_config

        # Run path and name for easy referral later
        self.run_path = f"{slt_config.entity_name}/{slt_config.wandb_project_name}"
        self.run_name = slt_config.run_name
        
        # Get run information
        self.api = wandb.Api()
        run_list = self.api.runs(
            path=self.run_path, 
            filters={
                "display_name": self.run_name,
                "state": "finished",
                },
            order="created_at", # Default descending order so backwards in time
        )
        assert run_list, f"Specified run {self.run_name} does not exist"
        self.run_api = run_list[slt_config.run_idx]
        
        try: self.history = self.run_api.history()
        except: self.history = self.run_api.history
        self.loss_history = self.history["Train Loss"]
        self.accuracy_history = self.history["Train Acc"]
        self.steps = self.history["_step"]

        # Get artifacts from old run
        self.state_queue = Queue()
        self.logits_queue = Queue()
        for artifact in self.run_api.logged_artifacts():
            if artifact.type == "states":
                self.state_queue.put(artifact)
            elif artifact.type == "logits_cp":
                self.logits_queue.put(artifact)
            

        # self.config: MainConfig = OmegaConf.create(self.run_api.config)
        self.config = self._get_config()
        
        # Set total number of unique samples seen (n). If this is not done it will break LLC estimator.
        self.slt_config.rlct_config.sgld_kwargs.num_samples = self.slt_config.rlct_config.num_samples = self.config.rlct_config.sgld_kwargs.num_samples
        self.slt_config.nano_gpt_config = self.config.nano_gpt_config
        
        # Log old config and SLT config to new run for post-analysis information
        self._set_logger()
        
        self.ed_loader = create_dataloader_hf(self.config, deterministic=True) # Make sure deterministic to see same data
        
        self.model, param_inf_properties = construct_model(self.config)
        
        # SLT stuff
        self.rlct_data_list: list[dict[str, float]] = []
        self.rlct_criterion = construct_rlct_criterion(self.config)
    
    
    def do_ed(self):
        """Main executable function of this class."""
        if self.slt_config.ed:
            ed_logits = self._get_ed_logits_from_checkpoints()
            ed_logits: list[torch.Tensor] = self._truncate_ed_logits(ed_logits)
            ed_projected_samples = self._ed_calculation(ed_logits)
    
            # Create and call instance of essential dynamics osculating circle plotter
            ed_plotter = EssentialDynamicsPlotter(ed_projected_samples, self.steps, self.slt_config.ed_plot_config, self.run_name)
            wandb.log({"ed_osculating": wandb.Image("ed_osculating_circles.png")})
    
    
    def plot_form_potential(self):
        """After analysing initial osculating circle plot, choose marked cusp data points.
        Use these to plot a form potential plot over time steps.
        """
        form_potential_plotter = FormPotentialPlotter(self.ed_logits, self.steps, self.slt_config, self.run_name)
        form_potential_plotter.plot()
    
    
    def _restore_states(self, idx: int) -> dict:
        """Restore model state from a checkpoint. Called once for every epoch.
        
        Params:
            idx: Index in steps.
            
        Returns:
            model state dictionary.
        """
        match self.config.model_save_method:
            case "wandb":
                artifact = self.run.use_artifact(f"{self.run_path}/states:idx{idx}_{self.run_name}_{self.config.time}")
                data_dir = artifact.download()
                model_state_path = Path(data_dir) / "states.torch"
                states = torch.load(model_state_path)
            case "aws":
                with s3.open(f"{self.config.aws_bucket}/{self.config.run_name}_{self.config.time}/{idx}") as f:
                    states = torch.load(f)
        return states["model"]


    def _restore_states_from_queue(self) -> dict:
        """Restore model state from a checkpoint. Called once for every epoch.
        
        Params:
            idx: Index in steps.
            
        Returns:
            model state dictionary.
        """
        match self.config.model_save_method:
            case "wandb":
                artifact = self.state_queue.get()
                data_dir = artifact.download()
                model_state_path = Path(data_dir) / "states.torch"
                states = torch.load(model_state_path)
            case "aws":
                with s3.open(f"{self.config.aws_bucket}/{self.config.run_name}_{self.config.time}/{idx}") as f:
                    states = torch.load(f)
        return states["model"]

    
    def _get_config(self) -> MainConfig:
        """"
        Manually get config from run as artifact. 
        WandB also logs automatically for each run, but it doesn't log enums correctly.
        """
        # artifact = artifact = self.api.artifact(f"{self.run_path}/states:dihedral_config_state") # Used as a test with manual artifact labelling
        artifact = self.api.artifact(f"{self.run_path}/config:{self.run_name}") # TODO: change to add time of run as well, but time needs to be gotten natively from WandB (chicken and egg)
        data_dir = artifact.download()
        config_path = Path(data_dir) / "config.yaml"
        return OmegaConf.load(config_path)
    
    
    # def _load_ed_logits(self) -> None:
    #     """Load ED logits from WandB.
    #     Deprecated method from when I used to save all logits in one go.
    #     """
    #     artifact = self.api.artifact(f"{self.run_path}/logits:{self.run_name}")
    #     data_dir = artifact.download()
    #     logit_path = Path(data_dir) / "logits_artifact"
    #     self.ed_logits = torch.load(logit_path)
    
    
    def _load_ed_logits(self) -> None:
        """Load ED logits from WandB when they are saved for each checkpoint separately.
        """
        ed_logits = []
        
        for checkpoint_idx in range(1, self.config.num_training_iter // self.config.rlct_config.ed_config.eval_frequency):
            idx = checkpoint_idx * self.config.rlct_config.ed_config.eval_frequency
            artifact = self.logits_queue.get()
            data_dir = artifact.download()
            logit_path = Path(data_dir) / f"logits_cp_{idx}"
            ed_logits.append(torch.load(logit_path))
            
        return ed_logits
    
    
    def _get_ed_logits_from_checkpoints(self) -> list[torch.Tensor]:
        """Based on current IO, need to load logits directly from WandB.
        """
        if self.config.use_logits:
            ed_logits = self._load_ed_logits()

        else:
            ed_logits = []
            
            for checkpoint_idx in range(1, self.config.num_training_iter // self.config.rlct_config.ed_config.eval_frequency):
                idx = checkpoint_idx * self.config.rlct_config.ed_config.eval_frequency
                state_dict = self._restore_states_from_queue(idx)
                self.model.load_state_dict(state_dict)
                
                logits_epoch = []
                with torch.no_grad():
                    for data in take_n(self.ed_loader, self.config.rlct_config.ed_config.batches_per_checkpoint):
                        inputs = data["input_ids"].to(self.device)
                        logits = self.model(inputs)
                        # Flatten over batch, class and sequence dimension
                        logits_epoch.append(rearrange(logits, 'b c s -> (b c s)'))
                
                # Concat all per-batch logits over batch dimension to form one super-batch
                self.logits_epoch = torch.cat(logits_epoch)
                # Append to binary file
                append_tensor_to_file(self.logits_epoch, self.logits_path)
                ed_logits.append(self.logits_epoch)
            
        return ed_logits
        
    
    def _truncate_ed_logits(self, ed_logits: list[torch.Tensor]) -> None:
        """Truncate run to clean out overtraining at end for cleaner ED plots.
        Determines the cutoff index for early stopping based on log loss.
        """
        # Manually specify cutoff index
        if self.slt_config.truncate_its is not None:
            total_its = len(self.loss_history) * self.config.eval_frequency
            ed_logit_cutoff_idx = len(ed_logits) * self.slt_config.truncate_its // total_its
            ed_logits = ed_logits[:ed_logit_cutoff_idx]
            return
        
        # Automatically calculate cutoff index using early-stop patience
        log_loss_values = np.log(self.loss_history.to_numpy())
        smoothed_log_loss = np.convolve(log_loss_values, np.ones(self.slt_config.early_stop_smoothing_window)/self.slt_config.early_stop_smoothing_window, mode='valid')

        increases = 0
        for i in range(1, len(smoothed_log_loss)):
            if smoothed_log_loss[i] > smoothed_log_loss[i-1]:
                increases += 1
                if increases >= self.config.early_stop_patience:
                    # Index where the increase trend starts
                    cutoff_idx = (i - self.slt_config.early_stop_patience + 1) * self.config.eval_frequency # Cutoff idx in loss step
                    ed_logit_cutoff_idx = cutoff_idx * self.config.rlct_config.ed_config.eval_frequency // self.config.eval_frequency
                    ed_logits = ed_logits[:ed_logit_cutoff_idx]
            else:
                increases = 0
        
        return ed_logits
        
    
    def _ed_calculation(self, ed_logits: list[torch.Tensor]) -> np.ndarray:
        """PCA and plot part of ED.
        
        Diplay top 3 components against each other and show fraction variance explained.
        Save projected PCA samples and PCA object in separate files for later plotting.
        """
        pca = PCA(n_components=3)
        pca.fit(ed_logits.cpu().numpy())
        
        # Projected coordinates for plotting purposes
        pca_projected_samples = np.empty((len(ed_logits), 3))
        for i, row in enumerate(ed_logits):
            logits_epoch = rearrange(row, 'n -> 1 n').cpu().numpy()
            projected_vector = pca.transform(logits_epoch)[0]
            pca_projected_samples[i] = projected_vector
        explained_variance = pca.explained_variance_ratio_
        
        plot_pca_plotly(pca_projected_samples[:,0], pca_projected_samples[:,1], pca_projected_samples[:,2], self.config)
        plot_explained_var(explained_variance)
        
        wandb.log({
            "ED_PCA_truncated": wandb.Image("PCA.png"),
            "explained_var_truncated": wandb.Image("pca_explained_var.png"),
        })
        
        pca_samples_file_path = f"pca_projected_samples_{self.config.run_name}.pkl"
        with open(pca_samples_file_path, 'wb') as f:
            pickle.dump(pca_projected_samples, f)
            
        pca_file_path = f"pca_{self.config.run_name}.pkl"
        with open(pca_file_path, 'wb') as f:
            pickle.dump(pca, f)
        
        return pca_projected_samples
        
    
    def calculate_rlct(self):
        """Estimate RLCTs from a set of checkpoints after training, plot and dump graph on WandB."""
        train_loader = create_dataloader_hf(self.config, deterministic=False)

        for epoch in range(1, self.config.num_epochs): # TODO: currently epochs coincides with evaluations, but may not always be true
            idx = epoch * self.config.eval_frequency
            state_dict = self._restore_states(idx)
            
            rlct_func = partial(
                estimate_learning_coeff_with_summary,
                loader=train_loader,
                criterion=self.rlct_criterion,
                main_config=self.slt_config,
                checkpoint=state_dict,
                device=self.device,
            )

            results, callback_names = rlct_func(
            sampling_method=rlct_class_map[self.slt_config.rlct_config.sampling_method], 
            optimizer_kwargs=self.slt_config.rlct_config.sgld_kwargs
            )
            # TODO: check these are the right objects
            prev_run_stats_kwargs = {"loss": self.loss_history[epoch - 1], "acc": self.accuracy_history[epoch - 1]}
            results_filtered = extract_and_save_rlct_data(results, callback_names, sampler_type=self.slt_config.rlct_config.sampling_method.lower(), idx=idx, kwargs=prev_run_stats_kwargs)
            self.rlct_data_list.append(results_filtered)
        
        rlct_df = pd.DataFrame(self.rlct_data_list)
        rlct_df.to_csv(f"rlct_{self.config.run_name}.csv")
        rlct_artifact = wandb.Artifact(f"rlct", type="rlct", description="All RLCT data.")
        rlct_artifact.add_file(f"rlct_{self.config.run_name}.csv")
        wandb.log_artifact(rlct_artifact, aliases=[f"rlct_{self.config.run_name}"])
    
    
    def _set_logger(self) -> None:
        """Call at initialisation to set loggers to WandB and/or AWS.
        Run naming convention is preprend 'post' to distinguish from training runs.
        """
        self.config["slt_config"] = self.slt_config # For saving to WandB
        # Add previous run id to tie runs together
        self.config["prev_run_path"] = f"{self.run_path}/{self.run_api.id}"
        logger_params = {
            "name": f"post_{self.config.run_name}",
            "project": self.config.wandb_config.wandb_project_name,
            "settings": wandb.Settings(start_method="thread"),
            "config": OmegaConf.to_container(self.config, resolve=True, throw_on_missing=True),
            "mode": "disabled" if not self.config.is_wandb_enabled else "online",
        }
        self.run = wandb.init(**logger_params, entity=self.config.wandb_config.entity_name)
        
        # Location on remote GPU of WandB cache to delete periodically
        self.wandb_cache_dirs = [Path.home() / ".cache/wandb/artifacts/obj", Path.home() / "root/.cache/wandb/artifacts/obj"]
        
        
    def finish_run(self):
        if self.config.is_wandb_enabled:
            wandb.finish()
        
            upload_cache_dir = Path.home() / "root/.local/share/wandb/artifacts/staging" 
            if upload_cache_dir.is_dir():
                shutil.rmtree(upload_cache_dir)
                
            time.sleep(60)
            shutil.rmtree("wandb")
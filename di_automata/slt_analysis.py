import wandb
from typing import TypeVar
from pathlib import Path
from functools import partial
import shutil
import time
import pickle
from einops import rearrange
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import subprocess
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
from di_automata.devinterp.ed_utils import EssentialDynamicsPlotter
Sweep = TypeVar("Sweep")


class PostRunSLT:
    def __init__(self, slt_config: PostRunSLTConfig):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.slt_config: PostRunSLTConfig = slt_config

        # Run path and name for easy referral later
        self.run_path = f"{slt_config.entity_name}/{slt_config.wandb_project_name}-alpha"
        self.run_name = slt_config.run_name
        
        # Get run information
        self.api = wandb.Api()
        run_list = self.api.runs(
            path=self.run_path, 
            filters={
                "display_name": self.run_name,
                # "$or": [{"state": "crashed"}, {"state": "finished"}],
                },
            order="created_at", # Default descending order so backwards in time
        )
        self.run_api = run_list[0]
        try: self.history = self.run_api.history()
        except: self.history = self.run_api.history
        self.loss_history = self.history["Train Loss"]
        self.accuracy_history = self.history["Train Acc"]
        self.steps = self.history["_step"]

        # Get full config logged for the run we are analysing off WandB
        # My logic for picking 100 here is it's probably going to exist, but this might cause errors
        try:
            self.config: MainConfig = self._get_config()
        except:
            self.config: MainConfig = self._get_config_old(idx=3800)
        
        # Set total number of unique samples seen (n). Be careful if this is not done it will break LLC estimator.
        self.slt_config.rlct_config.sgld_kwargs.num_samples = self.slt_config.rlct_config.num_samples = self.config.rlct_config.sgld_kwargs.num_samples
        self.slt_config.nano_gpt_config = self.config.nano_gpt_config
        
        # Now that you have config, log config again. Set new run to write RLCT information to
        self._set_logger()
        
        self.ed_loader = create_dataloader_hf(self.config, deterministic=True) # Make sure deterministic to see same data
        
        self.model, param_inf_properties = construct_model(self.config)
        
        self.rlct_data_list: list[dict[str, float]] = []
        self.rlct_criterion = construct_rlct_criterion(self.config)
        self.ed_logits = []
    
    
    def run_slt(self):
        """Main executable function of this class."""
        if self.slt_config.ed:
            self._load_ed_logits()
            self._truncate_ed_logits()
            ed_projected_samples = self._ed_calculation()
            
            # Create and call instance of essential dynamics osculating circle plotter
            ed_plotter = EssentialDynamicsPlotter(ed_projected_samples, self.steps, self.slt_config.ed_plot_config, self.run_name)
            wandb.log({"ed_osculating": wandb.Image("ed_osculating_circles.png")})

        if self.slt_config.llc:
            self._rlct()
    
    
    def _restore_states(self, idx: int) -> None:
        """Restore model state from a checkpoint. Called once for every epoch.
        
        Params:
            idx: Index in steps.
        """
        artifact = self.run.use_artifact(f"{self.run_path}/states:idx{idx}_{self.run_name}")
        data_dir = artifact.download()
        model_state_path = Path(data_dir) / "states.torch"
        states = torch.load(model_state_path)
        self.model_state_dict = states["model"]
    
    
    def _get_config_old(self, idx: int) -> MainConfig:
        """Restore config from a checkpoint. Only call once in entire run.
        
        Params:
            idx: Index in steps.
        """
        artifact = self.api.artifact(f"{self.run_path}/states:idx{idx}_{self.run_name}")
        data_dir = artifact.download()
        config_path = Path(data_dir) / "config.yaml"
        return OmegaConf.load(config_path)

    
    def _get_config(self) -> MainConfig:
        """"Newer version of above function which removes need for indexing by separating config saving from model checkpointing."""
        artifact = self.api.artifact(f"{self.run_path}/config:{self.run_name}")
        data_dir = artifact.download()
        config_path = Path(data_dir) / "config.yaml"
        return OmegaConf.load(config_path)
    
    
    def _load_ed_logits(self) -> None:
        """Load ED logits from WandB."""
        # artifact = self.api.artifact(f"{self.run_path}/logits:{self.run_name}")
        artifact = self.api.artifact(f"{self.run_path}/logits:dihedral_test")
        data_dir = artifact.download()
        logit_path = Path(data_dir) / "logits_artifact"
        self.ed_logits = torch.load(logit_path)
    
    
    def _truncate_ed_logits(self) -> None:
        """Truncate run to clean out overtraining at end for cleaner ED plots.
        Determines the cutoff index for early stopping based on log loss.
        """
        # Manually specify cutoff index
        if self.slt_config.truncate_its is not None:
            total_its = len(self.loss_history) * self.config.eval_frequency
            ed_logit_cutoff_idx = len(self.ed_logits) * self.slt_config.truncate_its // total_its
            self.ed_logits = self.ed_logits[:ed_logit_cutoff_idx]
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
                    self.ed_logits = self.ed_logits[:ed_logit_cutoff_idx]
            else:
                increases = 0
        
    
    def _ed_calculation(self) -> np.ndarray:
        """PCA and plot part of ED.
        
        Diplay top 3 components against each other and show fraction variance explained.
        """
        pca = PCA(n_components=3)
        pca.fit(self.ed_logits.cpu().numpy())
        
        # Projected coordinates for plotting purposes
        pca_projected_samples = np.empty((len(self.ed_logits), 3))
        for i, row in enumerate(self.ed_logits):
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

        return pca_projected_samples
                
    
    def _rlct(self):
        """Estimate RLCTs from a set of checkpoints after training, plot and dump graph on WandB."""
        train_loader = create_dataloader_hf(self.config, deterministic=False)

        for epoch in range(1, self.config.num_epochs): # TODO: currently epochs coincides with evaluations, but may not always be true
            idx = epoch * self.config.eval_frequency
            self._restore_states(idx)
            
            rlct_func = partial(
                estimate_learning_coeff_with_summary,
                loader=train_loader,
                criterion=self.rlct_criterion,
                main_config=self.slt_config,
                checkpoint=self.model_state_dict,
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
        
            # upload_cache_dir = Path.home() / "root/.local/share/wandb/artifacts/staging" 
            # if upload_cache_dir.is_dir():
            #     shutil.rmtree(upload_cache_dir)
                
            time.sleep(60)
            shutil.rmtree("wandb")
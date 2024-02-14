import wandb
from typing import TypeVar
from pathlib import Path
from functools import partial
import shutil
import time
from einops import rearrange
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

import torch

from di_automata.devinterp.optim.sgld import SGLD
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
Sweep = TypeVar("Sweep")


image_folder = rlct_folder = Path(__file__).parent / "images"


class PostRunSLT:
    def __init__(self, config: MainConfig):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.config = config
        
        self.ed_loader = create_dataloader_hf(self.config, deterministic=True) # Make sure deterministic to see same data
        
        self.model, param_inf_properties = construct_model(config)
        
        self.rlct_data_list: list[dict[str, float]] = []
        self.rlct_folder = Path(__file__).parent / self.config.rlct_config.rlct_data_dir
        self.rlct_folder.mkdir(parents=True, exist_ok=True)
        self.rlct_criterion = construct_rlct_criterion(self.config)
        
        self.ed_logits = []
        
        # Get run information
        self.api = wandb.Api()
        run_api = self.api.runs(
            path=f"{config.wandb_config.entity_name}/{config.wandb_config.wandb_project_name}", 
            filters={"display_name": {"$regex": f"^{config.run_name}$"} },
            order="created_at", # Default descending order
        )[0]
        wandb.init(
            entity=f"{config.wandb_config.entity_name}", 
            project=f"{config.wandb_config.wandb_project_name}", 
            id=run_api.id, 
            resume="must"
        )
        
        try: self.history = run_api.history()
        except: self.history = run_api.history
        self.loss_history = self.history["Train Loss"]
        self.accuracy_history = self.history["Train Acc"]
    
    
    def _ed(self):
        self._truncate_ed_logits()
        self._load_ed_logits()
        self._ed_calculation()
    
    
    def _restore_states(self, idx: int) -> None:
        """Restore model state from a checkpoint.
        Args:
            idx: Ideally index but can also be epoch. TODO: currently all saved on wandb as epochs.
            This is incorrect for ed calculations but correct for RLCT. Going ahead, all transition to idx.
        """
        artifact = self.run.use_artifact(f"{self.config.wandb_config.entity_name}/{self.config.wandb_config.wandb_project_name}/states:idx{idx}_{self.config.run_name}")
        data_dir = artifact.download()
        model_state_path = Path(data_dir) / "states.torch"
        states = torch.load(model_state_path)
        self.model_state_dict = states["model"]
    
    
    def _load_ed_logits(self) -> None:
        """Load ED logits from WandB."""
        artifact = self.api.artifact(f"{self.config.wandb_config.entity_name}/{self.config.wandb_config.wandb_project_name}/logits:{self.config.run_name}")
        logits = artifact.download()
        self.ed_logits = torch.load(logits)
    
    
    def _truncate_ed_logits(self) -> None:
        """Truncate run to clean out overtraining at end for cleaner ED plots.
        Determines the cutoff index for early stopping based on log loss.
        """
        log_loss_values = np.log(self.loss_history.to_numpy())
        smoothed_log_loss = np.convolve(log_loss_values, np.ones(self.config.early_stop_smoothing_window)/self.config.early_stop_smoothing_window, mode='valid')

        increases = 0
        for i in range(1, len(smoothed_log_loss)):
            if smoothed_log_loss[i] > smoothed_log_loss[i-1]:
                increases += 1
                if increases >= self.config.early_stop_patience:
                    # Index where the increase trend starts
                    cutoff_idx = i - self.config.early_stop_patience + 1
                    self.ed_logits = self.ed_logits[:cutoff_idx]
            else:
                increases = 0
        
    
    def _ed_calculation(self) -> None:
        """PCA and plot part of ED.
        
        Diplay top 3 components against each other and show fraction variance explained.
        """
        pca = PCA(n_components=3)
        concat_logit_matrix = torch.stack(self.ed_logits)
        pca.fit(concat_logit_matrix.cpu().numpy())
        
        projected_1, projected_2, projected_3 = [], [], []
        for row in self.ed_logits:
            logits_epoch = rearrange(row, 'n -> 1 n').cpu().numpy()
            projected_vector = pca.transform(logits_epoch)[0]
            projected_1.append(projected_vector[0])
            projected_2.append(projected_vector[1])
            projected_3.append(projected_vector[2])
        explained_variance = pca.explained_variance_ratio_
        
        plot_pca_plotly(projected_1, projected_2, projected_3, self.config)
        plot_explained_var(explained_variance)
        wandb.log({
            "ED_PCA": wandb.Image("PCA.png"),
            "explained_var": wandb.Image("pca_explained_var.png"),
        })
        
        torch.save(concat_logit_matrix, "logits")
        logit_artifact = wandb.Artifact(f"logits", type="logits", description="Logits across whole of training, stacked into matrix.")
        logit_artifact.add_file("logits", name="logits_artifact")
        wandb.log_artifact(logit_artifact, aliases=[f"{self.config.run_name}"])
        self._del_wandb_cache()
    
    
    def _rlct(self):
        """Estimate RLCTs from a set of checkpoints after training and dump on WandB.
        
        TODO: edit name of artifact from counting epochs to counting iterations.
        See function self._restore_states()
        """
        train_loader = create_dataloader_hf(self.config, deterministic=False)

        for epoch in range(self.config.num_epochs): # TODO: currently epochs coincides with evaluations, but may not always be true
            self._restore_states(epoch)
            
            rlct_func = partial(
                estimate_learning_coeff_with_summary,
                loader=train_loader,
                criterion=self.rlct_criterion,
                main_config=self.config,
                checkpoint=self.model_state_dict,
                device=self.device,
            )

            results, callback_names = rlct_func(
            sampling_method=rlct_class_map[self.config.rlct_config.sampling_method], 
            optimizer_kwargs=self.config.rlct_config.sgld_kwargs
            )
            results_filtered = extract_and_save_rlct_data(results, callback_names, sampler_type=self.config.rlct_config.sampling_method.lower(), idx=epoch*self.config.eval_frequency)
            self.rlct_data_list.append(results_filtered)
            
        rlct_df = pd.DataFrame(self.rlct_data_list)
        rlct_df.to_csv(rlct_folder / f"{self.config.run_name}.csv")
        rlct_artifact = wandb.Artifact(f"rlct", type="rlct", description="RLCT diagnostics.")
        rlct_artifact.add_file(rlct_folder / f"{self.config.run_name}.csv")
        wandb.log_artifact(rlct_artifact, aliases=[f"rlct_{self.config.run_name}"])
    
    
    def finish_analysis(self):
        wandb.finish()
        
        upload_cache_dir = Path.home() / "root/.local/share/wandb/artifacts/staging" 
        if upload_cache_dir.is_dir():
            shutil.rmtree(upload_cache_dir)
            
        time.sleep(60)
        shutil.rmtree("wandb")
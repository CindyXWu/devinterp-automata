import wandb
from typing import TypeVar
from pathlib import Path
from functools import partial
import shutil
import time
import pickle
import os
import s3fs
import re
from tqdm import tqdm
from einops import rearrange
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv.main import load_dotenv
from omegaconf import OmegaConf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
load_dotenv()
aws_key, aws_secret = os.getenv("AWS_KEY"), os.getenv("AWS_SECRET")
s3 = s3fs.S3FileSystem(key=aws_key, secret=aws_secret)

PATTERN = re.compile(r"logits_cp_(\d+):v")


def extract_number(artifact):
    """"
    Use for sorting artifacts so they can be properly queued.
    Split from the right to handle cases with multiple underscores.
    """
    match = PATTERN.search(artifact.name)
    return int(match.group(1))


class PostRunSLT:
    def __init__(self, slt_config: PostRunSLTConfig):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.slt_config: PostRunSLTConfig = slt_config

        # Run path and name for easy referral later
        self.run_path = f"{slt_config.entity_name}/{slt_config.wandb_project_name}"
        self.run_name = slt_config.run_name
        
        # Get run information
        self.api = wandb.Api(timeout=3000)
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
        self.time = self.run_api.config["time"]

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
        self.logits_file_path = Path(__file__).parent / f"logits/{self.config.run_name}_{self.config.time}"
        self.num_cps_to_analyse = self.steps.iloc[-1] // (self.config.rlct_config.ed_config.eval_frequency * self.slt_config.skip_cps)
        self.ed_logits = None

    
    def do_ed(self):
        """Main executable function of this class."""
        assert self.slt_config.ed, "Are you sure you want to go ahead and do ED?"

        # Choose whether to use logits directly from WandB run or do another inference pass on model states
        ed_logits = self._load_logits_cp() if self.slt_config.use_logits else self._load_logits_states()
        self.ed_logits: list[torch.Tensor] = self._truncate_ed_logits(ed_logits)
        ed_projected_samples = self._ed_calculation(self.ed_logits)

        # Create and call instance of essential dynamics osculating circle plotter
        if self.slt_config.osculating:
            steps = np.arange(0, self.ed_logits.shape[0])
            ed_plotter = EssentialDynamicsPlotter(ed_projected_samples, steps=steps, ed_plot_config=self.slt_config.ed_plot_config, run_name=self.run_name)
            wandb.log({"ed_osculating": wandb.Image("ed_osculating_circles.png")})
    
    
    def plot_form_potential(self):
        """After analysing initial osculating circle plot, choose marked cusp data points.
        Use these to plot a form potential plot over time steps.

        Should always be run after ed logit calculation for now.
        """
        if self.ed_logits is None: # Sometimes this code block comes after ed, so logits already loaded
            self.ed_logits = self._load_logits_cp() if self.slt_config.use_logits else self._load_logits_states()
        form_potential_plotter = FormPotentialPlotter(
            samples=self.ed_logits,
            steps=self.steps, 
            slt_config=self.slt_config,
            time=self.time,
        )
        form_potential_plotter.plot()

    
    def _get_config(self) -> MainConfig:
        """"
        Manually get config from run as artifact. 
        WandB also logs automatically for each run, but it doesn't log enums correctly.
        """
        artifact = self.api.artifact(f"{self.run_path}/config:{self.run_name}_{self.time}")
        data_dir = artifact.download()
        config_path = Path(data_dir) / "config.yaml"
        return OmegaConf.load(config_path)


    def _load_logits_cp(self) -> torch.Tensor:
        """Load logits from WandB for each checkpoint via multithreading.
        """
        if os.path.exists(self.logits_file_path):
            print(f"Loading existing logits from {self.logits_file_path}")
            ed_logits = torch.load(self.logits_file_path)
            print("Done loading existing logits")
        else:
            ed_logits = torch.zeros((self.num_cps_to_analyse, self.config.rlct_config.ed_config.batches_per_checkpoint * self.config.dataloader_config.train_bs * self.config.task_config.output_vocab_size * self.config.task_config.length))

            if self.config.model_save_method == "wandb":
                print("Getting WandB artifacts")
                logit_artifacts = [x for x in self.run_api.logged_artifacts() if x.type == "logits_cp"]
                logit_artifacts = sorted(logit_artifacts, key=extract_number)
                self.logit_artifacts = logit_artifacts[::self.slt_config.skip_cps]
            elif self.config.model_save_method == "aws":
                print("Getting AWS checkpoints")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self._load_logits_single_cp, cp_idx, ed_logits) for cp_idx in range(self.num_cps_to_analyse)]
                for future in tqdm(futures):
                    future.result()  # Wait for all futures to complete

            # assert check_no_zero_rows(ed_logits), "Error in loading ed logits and contains missing data"
            torch.save(ed_logits, self.logits_file_path)
        
        return ed_logits

    
    def _load_logits_single_cp(self, cp_idx: int, ed_logits: torch.Tensor) -> None:
        """Load just a single cp. 
        This function is designed to be called in multithreading and is called by the above function.
        """
        try:
            idx = cp_idx * self.config.rlct_config.ed_config.eval_frequency * self.slt_config.skip_cps
            match self.config.model_save_method:
                case "wandb":
                    artifact = self.logit_artifacts[cp_idx]
                    data_dir = artifact.download()
                    logit_path = Path(data_dir) / f"logits_cp_{idx}.torch"
                    ed_logits[cp_idx] = torch.load(logit_path)
                case "aws":
                    with s3.open(f'{self.config.aws_bucket}/{self.config.run_name}_{self.config.time}/logits_cp_{idx}.pth', mode='rb') as file:
                        ed_logits[cp_idx] = torch.load(file)
                    
        except Exception as e:
            print(f"Error fetching logits at step {idx}: {e}")

    
    def _restore_state_single_cp(self, cp_idx: int) -> dict:
        """Restore model state from a single checkpoint.
        Used in _load_logits_states() and _calculate_rlct().
        
        Args:
            idx_cp: index of checkpoint.
            
        Returns:
            model state dictionary.
        """
        idx = cp_idx * self.config.rlct_config.ed_config.eval_frequency
        match self.config.model_save_method:
            case "wandb":
                artifact = self.state_artifacts[cp_idx]
                data_dir = artifact.download()
                state_path = Path(data_dir) / f"states_{idx}.torch"
                state_dict = torch.load(state_path)
            case "aws":
                with s3.open(f'{self.config.aws_bucket}/{self.config.run_name}_{self.config.time}/states_{idx}.pth', mode='rb') as file:
                    state_dict = torch.load(file)
        return state_dict
    
    
    def _load_logits_states(self) -> torch.Tensor:
        """Load checkpointed states from WandB and do forward pass to get new logits.
        Only use for out of distribution evaluations where logits saved on WandB don't suffice.
        Uses multiprocess.
        """
        if os.path.exists(self.logits_file_path):
            ed_logits = torch.load(self.logits_file_path)
            print("Done loading existing logits")
        else:
            ed_logits = torch.zeros((self.num_cps_to_analyse, self.config.rlct_config.ed_config.batches_per_checkpoint * self.config.dataloader_config.train_bs * self.config.task_config.output_vocab_size * self.config.task_config.length))

            if self.config.model_save_method == "wandb":
                print("Getting WandB artifacts")
                state_artifacts = [x for x in self.run_api.logged_artifacts() if x.type == "states"]
                self.state_artifacts = sorted(state_artifacts, key=extract_number)
            else:
                print("Getting AWS checkpoints")

            with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(self._load_logits_states_single_cp, cp_idx, ed_logits, self.model, self.ed_loader) for cp_idx in range(self.num_cps_to_analyse)]
                    for future in tqdm(futures):
                        future.result()  # Wait for all futures to complete

            # assert check_no_zero_rows(ed_logits), "Error in loading ed logits and contains missing data"
            torch.save(ed_logits, self.logits_file_path)
            
        return ed_logits


    def _load_logits_states_single_cp(self, cp_idx: int, ed_logits: torch.Tensor, model: nn.Module, ed_loader: DataLoader) -> None:
        """Load just a single cp state and do inference to get logits.
        This function is designed to be called in multithreading and is called by the above function.
        """
        try:
            state_dict = self._restore_state_single_cp(cp_idx)
            model.load_state_dict(state_dict)
            
            logits_cp = []
            with torch.no_grad():
                for data in take_n(ed_loader, self.config.rlct_config.ed_config.batches_per_checkpoint):
                    inputs = data["input_ids"].to(self.device)
                    logits = model(inputs)
                    # Flatten over batch, class and sequence dimension
                    logits_cp.append(rearrange(logits, 'b c s -> (b c s)'))
            
            # Concat all per-batch logits over batch dimension to form one super-batch
            logits_cp = torch.cat(logits_cp)
            ed_logits[cp_idx] = logits_cp
            
        except Exception as e:
            print(f"Error fetching state dict at step {idx}: {e}")
        
    
    def _truncate_ed_logits(self, ed_logits: list[torch.Tensor]) -> None:
        """Truncate run to clean out overtraining at end for cleaner ED plots.
        Determines the cutoff index for early stopping based on log loss.
        """
        # Manually specify cutoff index
        if self.slt_config.truncate_its is not None:
            total_its = len(self.loss_history) * self.config.eval_frequency
            ed_logit_cutoff_idx = len(ed_logits) * self.slt_config.truncate_its // total_its
            ed_logits = ed_logits[:ed_logit_cutoff_idx]
            return ed_logits
        
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
        print("doing ed calculation")
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

        
        pca_samples_file_path = Path(__file__).parent / f"ed_data/pca_projected_samples_{self.config.run_name}_{self.time}"
        torch.save(pca_projected_samples, pca_samples_file_path)
        pca_file_path = Path(__file__).parent / f"ed_data/pca_{self.config.run_name}_{self.time}"
        torch.save(pca, pca_file_path)

        wandb.log({
            "ED_PCA_truncated": wandb.Image("PCA.png"),
            "explained_var_truncated": wandb.Image("pca_explained_var.png"),
        })
        
        return pca_projected_samples
    
        
    ## TODO: figure out how often you want to calculate RLCT and create indices to iterate over, and slice artifact list with
    # def calculate_rlct(self):
    #     """Estimate RLCTs from a set of checkpoints after training, plot and dump graph on WandB."""
    #     train_loader = create_dataloader_hf(self.config, deterministic=False)

    #     for epoch in range(1, self.config.num_epochs): # TODO: currently epochs coincides with evaluations, but may not always be true
    #         state_dict = self._restore_state_single_cp(idx)
            
    #         rlct_func = partial(
    #             estimate_learning_coeff_with_summary,
    #             loader=train_loader,
    #             criterion=self.rlct_criterion,
    #             main_config=self.slt_config,
    #             checkpoint=state_dict,
    #             device=self.device,
    #         )

    #         results, callback_names = rlct_func(
    #         sampling_method=rlct_class_map[self.slt_config.rlct_config.sampling_method], 
    #         optimizer_kwargs=self.slt_config.rlct_config.sgld_kwargs
    #         )
    #         # TODO: check these are the right objects
    #         prev_run_stats_kwargs = {"loss": self.loss_history[epoch - 1], "acc": self.accuracy_history[epoch - 1]}
    #         results_filtered = extract_and_save_rlct_data(results, callback_names, sampler_type=self.slt_config.rlct_config.sampling_method.lower(), idx=idx, kwargs=prev_run_stats_kwargs)
    #         self.rlct_data_list.append(results_filtered)
        
    #     rlct_df = pd.DataFrame(self.rlct_data_list)
    #     rlct_df.to_csv(f"rlct_{self.config.run_name}.csv")
    #     rlct_artifact = wandb.Artifact(f"rlct", type="rlct", description="All RLCT data.")
    #     rlct_artifact.add_file(f"rlct_{self.config.run_name}.csv")
    #     wandb.log_artifact(rlct_artifact, aliases=[f"rlct_{self.config.run_name}"])
    
    
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


def check_no_zero_rows(tensor: torch.Tensor) -> bool:
    """Check if there's any non-zero element in each row of a torch tensor.
    Return true if there is at least one non-zero element in each row.
    """
    # torch.any returns True if there's at least one non-zero element in the row
    not_zero_rows = torch.any(tensor != 0, dim=1)
    # Ensure all rows have at least one non-zero element
    return torch.all(not_zero_rows)
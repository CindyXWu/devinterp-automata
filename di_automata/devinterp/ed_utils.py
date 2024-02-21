from typing import Union, List, Tuple
import os
from dotenv import load_dotenv
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import tqdm
import itertools
import pickle
import scipy.ndimage
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.gridspec as gridspec
import seaborn as sns
import wandb

from sklearn.decomposition import PCA
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from di_automata.devinfra.utils.iterables import int_linspace
from di_automata.config_setup import EDPlotConfig

logging.basicConfig(level=logging.INFO)
load_dotenv();

# Matplotlib defaults
fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)

sns.set_style("ticks")
# plt.rcParams["font.family"] = "Times New Roman"
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)
plt.rc('legend', fontsize=10)
plt.rc('axes', titlesize=12)
plt.rcParams['figure.dpi'] = 300

golden_ratio = (5**0.5 - 1) / 2
WIDTH = 3.25
HEIGHT = WIDTH * golden_ratio


def get_nearest_step(steps, step):
    """Used for osculating circle plots."""
    idx = np.argmin(np.abs(np.array(steps) - step))
    return steps[idx]
    

def osculating_circle(curve: List[np.ndarray], t_index: int) -> Tuple[float, float]:
    """Calculate radius and centre of curvature given a list of 2D/3D coordinates defining a curve."""
    # Handle edge cases
    if t_index == 0:
        t_index = 1
    if t_index == len(curve) - 1:
        t_index = len(curve) - 2

    # Central differences for first and second derivatives
    r_prime = (curve[t_index + 1] - curve[t_index - 1]) / 2
    r_double_prime = (curve[t_index + 1] - 2 * curve[t_index] + curve[t_index - 1])

    # Append a zero for 3D cross product
    r_prime_3d = np.append(r_prime, [0])
    r_double_prime_3d = np.append(r_double_prime, [0])
    
    # Curvature calculation and normal vector direction
    cross_product = np.cross(r_prime_3d, r_double_prime_3d)
    curvature = np.linalg.norm(cross_product) / np.linalg.norm(r_prime)**3
    signed_curvature = np.sign(cross_product[2])  # Sign of z-component of cross product
    radius_of_curvature = 1 / (curvature + 1e-12)
    
    # Unit tangent vector
    tangent = r_prime / np.linalg.norm(r_prime)

    # Unit normal vector, direction depends on the sign of the curvature
    if signed_curvature >= 0:
        norm_perp = np.array([-tangent[1], tangent[0]])  # Rotate tangent by 90 degrees counter-clockwise
    else:
        norm_perp = np.array([tangent[1], -tangent[0]])  # Rotate tangent by 90 degrees clockwise
    
    # Center of the osculating circle
    center = curve[t_index] + radius_of_curvature * norm_perp

    return center, radius_of_curvature


class EssentialDynamicsPlotter:
    def __init__(
        self, 
        samples: np.ndarray,
        steps: np.ndarray,
        ed_plot_config: EDPlotConfig,
        run_name: str,
    ):
        """
        Class for plotting osculating circles and caustic cusps.
        
        Args:
            samples: numpy array [n_samples, n_principle_features].
            steps: numpy array.
            ed_plot_config: OmegaConf object with same structure as EDPlotConfig.
        """
        self.config = ed_plot_config
        self.samples = samples
        print("Number of samples: " + str(len(samples)))
        self.steps = steps
        self.run_name = run_name
        
        ## Set constants
        # Start and end points for linear interpolation used for smoothing
        self.START_LERP = 0.2 * self.config.smoothing_late_boundary
        self.END_LERP = self.config.smoothing_late_boundary
        # Osculation calculation start and end
        self.CUTOFF_START = self.config.osculate_start
        self.CUTOFF_END = self.config.osculate_end_offset
        self.OSCULATE_SKIP = self.config.osculate_skip
        # Often-used range list for plotting marks
        self.PLOT_RANGE = range(self.CUTOFF_START, len(self.samples) - self.CUTOFF_END, self.OSCULATE_SKIP)

        # Make folder which belongs to particular run name
        self.ed_folder_path = Path(__file__).parent / f"{self.config.ed_folder}/{self.run_name}"
        self.ed_folder_path.mkdir(parents=True, exist_ok=True)
        
        self.I = 0
        
        # Set axes, matplotlib defaults etc
        self.fig, self.axes = self._set_matplotlib()
        self._smooth_all_components()
        # Body, grunt work of plotting
        self._plot_osculating_main()
        # Produce figure
        self._finish_plot()
        
        
    def _smooth_all_components(self) -> list[np.ndarray]:
        """Create and update self.smoothed_pcs: list of np.ndarray coordinates.
        Save to file via pickle.
        """
        self.smoothed_pcs = []
        
        for i in range(self.config.num_pca_components):
            file_path_smoothing = self.ed_folder_path / 'smoothed_principle_component_{i}.pkl'
            
            if self.config.use_cache and os.path.exists(file_path_smoothing):
                with open(file_path_smoothing, 'rb') as file:
                    smoothed_pc = pickle.load(file)
            else:
                smoothed_pc = self._smooth_component(i)
                
                with open(file_path_smoothing, 'wb') as file:
                    pickle.dump(smoothed_pc, file)
                    
            self.smoothed_pcs.append(smoothed_pc)


    def _smooth_component(self, i) -> np.ndarray:
        """Smooth single PCA component.
        smoothed_pc has same dimensions as one column of self.samples.

        Args:
            i: PCA component index.
        """
        smoothed_pc = np.zeros_like(self.samples[:,0])
        
        for t_idx in range(len(self.samples)):
            if t_idx < self.START_LERP:
                sigma = self.config.smoothing_sigma_early
            elif t_idx > self.END_LERP:
                sigma = self.config.smoothing_sigma_late
            else: # Linear interpolation
                sigma = (self.config.smoothing_sigma_late - self.config.smoothing_sigma_early)/(self.END_LERP - self.START_LERP) * (t_idx - self.START_LERP) + self.config.smoothing_sigma_early
                
            smoothed_pc[t_idx] = scipy.ndimage.gaussian_filter1d(self.samples[:, i], sigma)[t_idx]
            
        return smoothed_pc


    def _plot_osculating_main(self):
        """For each principle component pair we first do some data processing, then plotting.
        Calls multiple helper plotter functions below.
        """
        for i,j in itertools.combinations(range(self.config.num_pca_components), 2):
            print(f'Processing PC{j+1} vs PC{i+1}')

            smoothed_pc_i = self.smoothed_pcs[i]
            smoothed_pc_j = self.smoothed_pcs[j]
            self.smoothed_samples = np.column_stack((smoothed_pc_i, smoothed_pc_j))
            
            # Load osculating data
            file_path_osculating = self.ed_folder_path / f'osculating_data_i{i}_j{j}.pkl'
            if self.config.use_cache and os.path.exists(file_path_osculating):
                print("Using cached osculate data")
                with open(file_path_osculating, 'rb') as file:
                    osculating_data = pickle.load(file)
            else:
                osculating_data: dict[str, tuple[float, float]] = {}

                for t_idx in range(self.CUTOFF_START, len(self.samples) - self.CUTOFF_END, 1):
                    osculating_data[t_idx] = osculating_circle(self.smoothed_samples, t_idx)
                
                with open(file_path_osculating, 'wb') as file:
                    pickle.dump(osculating_data, file)
            
            self._plot_osculating_circles(osculating_data)
            self._draw_phases(i, j)
            # # This was needed in the original code because it didn't use itertools, but keep for safety measure
            # if self.I >= len(self.axes): 
            #     print("Continuing as beyond axes count")
            #     continue
            self._mark_points(osculating_data)
            
            self.axes[self.I].set_xlabel(f'PC {i+1}')
            self.axes[self.I].set_ylabel(f'PC {j+1}')

            self.I += 1


    def _plot_osculating_circles(self, osculating_data:  dict[str, tuple[float, float]]):
        """
        Args:
            osculating_data:  Value is tuple of radius and centre of curvature.
        """
        self.dcenter: dict[int, float] = {} # Distance between neighbouring centres, key by checkpoint idx
        self.radii: dict[int, float] = {} # Circle radii, key by checkpoint idx
        prev_center = None

        for t_idx in self.PLOT_RANGE:
            center, radius = osculating_data[t_idx]
            self.radii[t_idx] = radius
            self.dcenter[t_idx] = 1000 # Not sure why Dan set this, but I guess it's currently a large anchor value

            if prev_center is not None:
                d = np.linalg.norm(center - prev_center)
                self.dcenter[t_idx] = d

            color = 'lightgray'
            circle = plt.Circle((center[0].item(), center[1].item()), radius.item(), alpha=0.5, color=color, lw = 0.5, fill=False)

            # if self.I < len(self.axes):
            #     self.axes[self.I].add_artist(circle)
            self.axes[self.I].add_artist(circle)

            prev_center = center

        # Find high curvature points
        sorted_radii_list = sorted(list(self.radii.values()))
        self.sharp_radius_upper_bound = 0
        if len(sorted_radii_list) > self.config.num_sharp_points:
            # Radius determines a sharp point (i.e. small radii), note sorting ascending
            self.sharp_radius_upper_bound = sorted_radii_list[self.config.num_sharp_points]
        
        # Find caustic cusps
        sorted_dcenter_list = sorted(list(self.dcenter.values()))
        self.dcenter_bound = 0
        if len(sorted_dcenter_list) > self.config.num_vertices:
            self.dcenter_bound = sorted_dcenter_list[self.config.num_vertices]
        
        # Plot un-smoothed points in the background
        #if I < len(axes):
        #    axes[I].scatter(x=samples[:, i], y=samples[:, j], alpha=0.8, color="lightgray", s=10)
        
        
    def _draw_phases(self, i: int, j: int):
        """Colour each phase on the ED plot with a different line."""
        # if self.I < len(self.axes):
        # If we want to plot phases as separate sections
        if self.config.transitions:
            for k, (start, end, stage) in enumerate(self.config.transitions):
                start_idx  = self.steps.index(get_nearest_step(self.steps, start))
                end_idx = self.steps.index(get_nearest_step(self.steps, end)) + 1
                
                self.axes[self.I].plot(
                    self.smoothed_samples[start_idx :end_idx, 0], 
                    self.smoothed_samples[start_idx :end_idx, 1], 
                    color=self.colors[k], 
                    lw = 2
                )
        else:
            self.axes[self.I].plot(self.samples[:, i], self.samples[:, j])

        # Look for points where distance between neighbouring centres is small i.e. curve is relatively tight
        for t_idx in self.PLOT_RANGE:
            if t_idx > 2 * self.OSCULATE_SKIP and self.dcenter[t_idx] < self.dcenter_bound and self.dcenter[t_idx - self.OSCULATE_SKIP] < self.dcenter_bound:
                print("    Vertex [" + str(t_idx) + "] rate of change of osculate center " + str(self.dcenter[t_idx]))
                if self.I < len(self.axes):
                    self.axes[self.I].scatter(self.smoothed_samples[t_idx, 0], self.smoothed_samples[t_idx, 1], color='gold')

        # Mark in red high curvature points, skipping every self.OSCULATE_SKIP points
        for t_idx in self.PLOT_RANGE:
            if self.radii[t_idx] < self.sharp_radius_upper_bound:
                print("Sharp point [" + str(t_idx) + "] curvature " + str(self.radii[t_idx]))
                if self.I < len(self.axes):
                    self.axes[self.I].scatter(self.smoothed_samples[t_idx, 0], self.smoothed_samples[t_idx, 1], color='red')

    
    def _mark_points(self, osculating_data: dict[str, tuple[float, float]]):
        """
        Args:
            osculating_data: key checkpoint index, value tuple(radii,centres) of osculating circles.
        """
        current_x_limits = self.axes[self.I].get_xlim()
        current_y_limits = self.axes[self.I].get_ylim()
        
        # Draw the evolute
        if self.config.plot_caustic:
            for t_idx in range(self.CUTOFF_START, len(self.samples) - self.CUTOFF_END, 1):
                center, radius = osculating_data[t_idx]
                
                if center[0].item() > current_x_limits[0] and center[0].item() < current_x_limits[1]:
                    if center[1].item() > current_y_limits[0] and center[1].item() < current_y_limits[1]:
                        self.axes[self.I].scatter([center[0].item()], [center[1].item()], color='black', s=0.2)
                            
        if self.config.marked_cusp_data:
            for marked_cusp_id in range(len(self.config.marked_cusp_data)):
                marked_cusp = self.config.marked_cusp_data[marked_cusp_id]["step"]
                self.axes[self.I].scatter(
                    self.smoothed_samples[marked_cusp,0], 
                    self.smoothed_samples[marked_cusp,1], color='green', 
                    marker='x', 
                    s=40
                )
                center, radius = osculating_data[marked_cusp]
                self.axes[self.I].scatter(
                    [center[0].item()],
                    [center[1].item()], 
                    color='green', 
                    marker='x', 
                    s=40
                )

                if self.show_vertex_influence:
                    vertex_influence_start = self.marked_cusp_data[marked_cusp_id]["influence_start"]
                    vertex_influence_end = self.marked_cusp_data[marked_cusp_id]["influence_end"]         
                    self.axes[self.I].scatter(
                        self.smoothed_samples[vertex_influence_start,0], 
                        self.smoothed_samples[vertex_influence_start,1], 
                        color='blue', 
                        marker='x', 
                        s=40
                    )
                    self.axes[self.I].scatter(
                        self.smoothed_samples[vertex_influence_end,0], 
                        self.smoothed_samples[vertex_influence_end,1], 
                        color='blue', 
                        marker='x', 
                        s=40
                    )


    def _finish_plot(self):
        labels = [""]
        self.axes[0].set_ylabel(f"{labels[0]}\n\nPC 1")
        plt.tight_layout(rect=[0, 0, 1, 1])

        if self.config.transitions:
            legend_ax = self.fig.add_axes([0.1, -0.03, 0.95, 0.05])  # Adjust these values as needed
            handles = [plt.Line2D([0], [0], color=self.colors[i], linestyle='-') for i in range(len(self.config.transitions))]
            labels = [label for _, _, label in self.config.transitions]
            legend_ax.legend(handles, labels, loc='center', ncol=len(labels), frameon=False)
            legend_ax.axis('off')
        
        self.fig.set_facecolor('white')
        self.fig.savefig('ed_osculating_circles.png', dpi=300)


    def _set_matplotlib(self) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]]:
        """Set matplotlib defaults."""
        sns.set_style("ticks")
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('axes', labelsize=10)
        plt.rc('legend', fontsize=10)
        plt.rc('axes', titlesize=12)
        plt.rcParams['figure.dpi'] = 300
        
        golden_ratio = (5**0.5 - 1) / 2
        self.WIDTH = 3.25
        self.HEIGHT = WIDTH * golden_ratio
        
        self.palette = 'tab10'
        self.colors = self.config.colors or sns.color_palette(self.palette)
        
        self.fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
        
        num_pca_combos = (self.config.num_pca_components * (self.config.num_pca_components - 1)) // 2 # n choose 2

        fig, axes = plt.subplots(1, num_pca_combos, figsize=self.config.figsize)
        if num_pca_combos == 1:
            axes = [axes]
        
        return fig, axes


if __name__ == "__main__":
    """TODO: refactor"""
    # Example of how to use this class
    config_path = 'path/to/config.yaml'
    plotter = EssentialDynamicsPlotter(config_path)
    fig = plotter.plot()
    plt.show()
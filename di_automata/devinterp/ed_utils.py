from typing import Union, List, Tuple, TypeVar
import os
import logging
from pathlib import Path
import numpy as np
import itertools
import pickle
import scipy.ndimage
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns

from di_automata.config_setup import *

# Typing
T = TypeVar("T", bound=np.generic, covariant=True)
Vector = np.ndarray[Tuple[int], np.dtype[T]]
Matrix = np.ndarray[Tuple[int, int], np.dtype[T]]
Tensor = np.ndarray[Tuple[int, ...], np.dtype[T]]

logging.basicConfig(level=logging.INFO)

# Plotting globals
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
        self.ed_folder_path = Path(__file__).parent.parent / Path(f"{self.config.ed_folder}/{self.run_name}")
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
        self.smoothed_pcs: list[Vector] = []
        
        for i in range(self.config.num_pca_components):
            file_path_smoothing = self.ed_folder_path / f'smoothed_pc_{i}.pkl'
            
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
            file_path_osculating = Path(__file__).parent.parent / f"{self.config.ed_folder}/{self.run_name}"/f'osculating_i{i}_j{j}.pkl'
            if self.config.use_cache and os.path.exists(file_path_osculating):
                print("Using cached osculate data")
                with open(file_path_osculating, 'rb') as file:
                    osculating_data = pickle.load(file)
            else:
                print("Calculating osculating circles")
                osculating_data: dict[str, tuple[float, float]] = {}

                for t_idx in range(self.CUTOFF_START, len(self.samples) - self.CUTOFF_END, 1):
                    osculating_data[t_idx] = osculating_circle(self.smoothed_samples, t_idx)
                
                with open(file_path_osculating, 'wb') as file:
                    pickle.dump(osculating_data, file)
            
            self._plot_osculating_circles(osculating_data)
            self._draw_phases(i, j)
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
        # axes[I].scatter(x=samples[:, i], y=samples[:, j], alpha=0.8, color="lightgray", s=10)
        
        
    def _draw_phases(self, i: int, j: int):
        """Colour each phase on the ED plot with a different line. 
        Should be called inside a loop for i,j and with a particular self.I state.
        """
        print("Marking phases")
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
            # Plot all the points as a scruffy blue line
            # self.axes[self.I].plot(self.samples[:, i], self.samples[:, j], linewidth=0.5)
            # Nicer version: smoothed samples
            self.axes[self.I].plot(self.smoothed_samples[:, 0], self.smoothed_samples[:, 1])
            pass

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

                if self.config.show_vertex_influence:
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
        plt.tight_layout(rect=[0, 0, 1, 1])

        if self.config.transitions:
            legend_ax = self.fig.add_axes([0.1, -0.03, 0.95, 0.05])
            handles = [plt.Line2D([0], [0], color=self.colors[i], linestyle='-') for i in range(len(self.config.transitions))]
            labels = [label for _, _, label in self.config.transitions]
            legend_ax.legend(handles, labels, loc='center', ncol=len(labels), frameon=False)
            legend_ax.axis('off')
        
        self.fig.set_facecolor('white')
        self.fig.savefig('ed_osculating_circles.png', dpi=300)


    def _set_matplotlib(self) -> Tuple[matplotlib.figure.Figure, Union[matplotlib.axes.Axes, List[matplotlib.axes.Axes]]]:
        """Set matplotlib defaults."""
        self.palette = 'tab10'
        self.colors = self.config.colors or sns.color_palette(self.palette)
        self.fonts = set(f.name for f in matplotlib.font_manager.fontManager.ttflist)
        
        num_pca_combos = (self.config.num_pca_components * (self.config.num_pca_components - 1)) // 2 # n choose 2

        fig, axes = plt.subplots(1, num_pca_combos, figsize=self.config.figsize)
        if num_pca_combos == 1:
            axes = [axes]
        
        return fig, axes
    
    
class FormPotentialPlotter:
    """Take identified cusps and plot.
    """
    def __init__(
        self,
        samples: np.ndarray,
        steps: np.ndarray,
        slt_config: PostRunSLTConfig,
        time: str,
    ):
        """"
        This function requires marked cusp data to be present. It should be called after the initial osculating circle plot has been examined.
        
        Args:
            original_samples: unprojected original logit samples.
            steps: steps of samples as fed in (not truncated, but should be based on config influence_start and influence_end points).
        """
        self.samples = samples
        self.steps = steps
        self.slt_config = slt_config
        self.config = slt_config.ed_plot_config
        self.time = time
        self.marked_cusp_data = self.config.marked_cusp_data

        self.alpha = 1
        self.cusp_functions = []
        
        # Change only if you want more than 3 PCA components
        self.pc_pairs = [[0,1], [0,2], [1,2]]
        
        self._set_matplotlib()
        print("done initialising form potential plotter")

        # Various folders
        self.ed_folder_path = Path(__file__).parent.parent / f"ed_data/{self.slt_config.run_name}"
        self.pca_file_path = Path(__file__).parent.parent / f"ed_data/pca_{self.slt_config.run_name}_{self.time}"
        self.f_cusp_file_path = Path(__file__).parent.parent / f"forms/{self.slt_config.run_name}"
        
        
    def _set_matplotlib(self) -> None:
        plt.figure(figsize=(5, 3))
        plt.xlabel(r'Step $t$')
        plt.ylabel(r'$H_\alpha(w_t)$')
    
    
    def plot(self):
        print("plotting in form potential function")
        cusp_functions = []
        alpha = 1 # Index of form
        
        if os.path.exists(self.pca_file_path):
            pca = torch.load(self.pca_file_path)
        else:
            print(f"No pca found at {self.pca_file_path}")
            
        for cusp in self.marked_cusp_data:
            # Unpack cusp
            cusp_index = cusp["step"]
            influence_start = cusp["influence_start"]
            influence_end = cusp["influence_end"]
            sample_indices = np.arange(int(influence_start * 0.6), min(int(influence_end * 1.5), 1249))
            # sample_indices = np.arange(1, len(self.samples))
        
            print(f'Processing cusp at {cusp_index}')
            
            center_list = []
            
            for i, j in self.pc_pairs:
                # file_path_smoothing_1 = f'smoothed_pc_{i}.pkl'
                # file_path_smoothing_2 = f'smoothed_pc_{i}.pkl'
                # assert os.path.exists(file_path_smoothings), "No smoothing stored on disk"        
                # with open(file_path_smoothings, 'rb') as file:
                #     smoothed_samples = pickle.load(file)
                
                file_path_osculating = self.ed_folder_path / f'osculating_i{i}_j{j}.pkl'
                assert os.path.exists(file_path_osculating), f"No osculating data stored to disk at {file_path_osculating}"
                with open(file_path_osculating, 'rb') as file:
                    osculating_data = pickle.load(file)

                center, radius = osculating_data[cusp_index]
                center_list.append(center)

            # Want center_list[0] = [pc1, pc2]
            coeff_pc1 = center_list[0][0] = center_list[1][0]
            coeff_pc2 = center_list[0][1]
            # Want center_list[1] = [pc1, pc3]
            coeff_pc3 = center_list[1][1]
            
            # pca_vectors[t,:] is the (t+1)st principal component
            pca_vectors = pca.components_
            
            # Get cusp functions
            if self.config.num_pca_components == 3:
                f_cusp = pca_vectors[0,:] * coeff_pc1 + pca_vectors[1,:] * coeff_pc2 + pca_vectors[2,:] * coeff_pc3
                f_cusp_top = np.array([coeff_pc1, coeff_pc2, coeff_pc3])
            cusp_functions.append(f_cusp)

            # SAVE FORM LOGITS - VERY IMPORTANT
            with open(self.f_cusp_file_path, 'wb') as f:
                pickle.dump(f_cusp_top, f)

            # Project back for a sanity check
            I = 0
            for i, j in self.pc_pairs:
                components_to_use = [i,j]
                selected_components = pca.components_[components_to_use]
                f_projected = f_cusp.dot(selected_components.T)
                print(f"Norm distance for pair [{i},{j}] is {np.linalg.norm(f_projected - center_list[I])}")
                I += 1
                    
            distances = np.linalg.norm(self.samples[sample_indices,:self.config.num_pca_components] - f_cusp_top,axis=1)

            print(f"Distances length {len(distances)}")
            print(f"Principal components of sample at {cusp_index}: {self.samples[cusp_index,:4]}")
            print(f"Principal components of inferred cusp: {f_cusp_top}")

            distances = distances ** 2
            smoothed_distances = scipy.ndimage.gaussian_filter1d(distances, 1)
            distances = distances - np.min(smoothed_distances)
            smoothed_distances = smoothed_distances - np.min(smoothed_distances)
            
            # Smoothed distances are actually distance to 
            # plt.scatter(sample_indices * 100, distances, s=1, alpha=0.01)
            plt.plot(sample_indices * self.slt_config.rlct_config.ed_config.eval_frequency, smoothed_distances, lw=0.8, label=r'$\alpha = $' + str(alpha))
            
            alpha += 1
        
        self._finish_plot()
    
    
    def _finish_plot(self):
        plt.legend(loc='lower left')
        for marked_cusp in self.config.marked_cusp_data:
            plt.axvline(x=marked_cusp["influence_start"] * self.slt_config.rlct_config.ed_config.eval_frequency, color='black', linestyle=':', lw=1, alpha=0.1)
            plt.axvline(x=marked_cusp["influence_end"] * self.slt_config.rlct_config.ed_config.eval_frequency, color='black', linestyle=':', lw=1, alpha=0.1)
        plt.xscale('log')
        plt.savefig("form_potential.png", dpi=300)
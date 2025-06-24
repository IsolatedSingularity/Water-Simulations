"""
Static Visual Analysis of Water Simulation Models

This script generates and visualizes data from two different 2D water simulation
methods: Smoothed Particle Hydrodynamics (SPH) and a grid-based Stable Fluids
solver. It creates a comparative visualization of key fluid properties like
density, pressure, and velocity divergence for a "dam break" scenario.

The visualizations adhere to the project's quality standards, using specified
color palettes and clear, publication-quality labels with LaTeX rendering.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- General Simulation Parameters ---
GRID_SIZE = 64  # Resolution for grid-based method and for visualizing SPH
PARTICLES_PER_ROW = 20  # For SPH initialization
SIMULATION_STEPS = 50  # Number of steps to run for the snapshot
DT = 0.04  # Time step
GRAVITY = np.array([0, -9.8])

# --- SPH Parameters ---
PARTICLE_MASS = 1.0
GAS_CONST = 2000.0  # Gas constant for Ideal Gas State Equation
REST_DENSITY = 300.0  # Rest density
VISCOSITY = 200.0  # Viscosity constant
SMOOTHING_RADIUS = 0.8  # Radius for kernel smoothing (h)

# --- Plotting Parameters ---
PLOT_DIR = 'Plots'
OUTPUT_FILENAME = os.path.join(PLOT_DIR, 'static_comparison.png')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
# Use one of the approved color palettes
DIVERGING_CMAP = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
SEQUENTIAL_CMAP = sns.color_palette("mako", as_cmap=True)


# --- SPH Implementation ---

class SPHSolver:
    """A simple 2D SPH solver for the dam break scenario."""
    def __init__(self):
        # Initialize particles in a block on the left
        x = np.linspace(0, GRID_SIZE * 0.4, PARTICLES_PER_ROW)
        y = np.linspace(0, GRID_SIZE * 0.8, PARTICLES_PER_ROW * 2)
        xx, yy = np.meshgrid(x, y)
        positions = np.vstack([xx.ravel(), yy.ravel()]).T
        
        self.n_particles = len(positions)
        self.pos = positions.astype(float)
        self.vel = np.zeros_like(self.pos)
        self.acc = np.zeros_like(self.pos)
        self.rho = np.zeros(self.n_particles)
        self.pressure = np.zeros(self.n_particles)
        self.mass = np.full(self.n_particles, PARTICLE_MASS)

    # Kernel functions (Poly6 and Spiky)
    def _poly6_kernel(self, r_sq, h):
        coeff = 315.0 / (64.0 * np.pi * h**9)
        return coeff * (h**2 - r_sq)**3

    def _spiky_grad_kernel(self, r, h):
        coeff = -45.0 / (np.pi * h**6)
        return coeff * (h - r)**2

    def _viscosity_laplacian_kernel(self, r, h):
        coeff = 45.0 / (np.pi * h**6)
        return coeff * (h - r)

    def _update_fluid_properties(self):
        h = SMOOTHING_RADIUS
        for i in range(self.n_particles):
            density = 0.0
            for j in range(self.n_particles):
                r_vec = self.pos[i] - self.pos[j]
                r_sq = np.dot(r_vec, r_vec)
                if r_sq < h**2:
                    density += self.mass[j] * self._poly6_kernel(r_sq, h)
            self.rho[i] = density
            self.pressure[i] = GAS_CONST * (self.rho[i] - REST_DENSITY)

    def _compute_forces(self):
        h = SMOOTHING_RADIUS
        f_press = np.zeros_like(self.pos)
        f_visc = np.zeros_like(self.pos)

        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i == j:
                    continue
                r_vec = self.pos[i] - self.pos[j]
                r = np.linalg.norm(r_vec)
                if r < h:
                    if r > 1e-6: # Check for non-zero distance to avoid division by zero
                        # Pressure force
                        avg_pressure = (self.pressure[i] + self.pressure[j]) / 2.0
                        grad = self._spiky_grad_kernel(r, h) * (r_vec / r)
                        f_press[i] += -self.mass[j] * (avg_pressure / self.rho[j]) * grad
                        
                        # Viscosity force
                        laplacian = self._viscosity_laplacian_kernel(r, h)
                        f_visc[i] += VISCOSITY * self.mass[j] * ((self.vel[j] - self.vel[i]) / self.rho[j]) * laplacian
        
        self.acc = (f_press + f_visc) / self.rho[:, np.newaxis] + GRAVITY

    def _integrate(self):
        # Verlet integration
        self.vel += self.acc * DT
        self.pos += self.vel * DT
        
        # Enforce boundaries
        self.pos = np.clip(self.pos, 0, GRID_SIZE - 1)

    def step(self):
        """Perform one step of the SPH simulation."""
        self._update_fluid_properties()
        self._compute_forces()
        self._integrate()

    def get_gridded_data(self):
        """Interpolate particle data onto a grid."""
        grid = np.arange(GRID_SIZE)
        x_grid, y_grid = np.meshgrid(grid, grid)
        
        density_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        pressure_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        vel_div_grid = np.zeros((GRID_SIZE, GRID_SIZE))
        
        # Simplified interpolation: average properties of particles in each cell
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                indices = np.where((np.floor(self.pos[:,0]) == i) & (np.floor(self.pos[:,1]) == j))[0]
                if len(indices) > 0:
                    density_grid[j, i] = np.mean(self.rho[indices])
                    pressure_grid[j, i] = np.mean(self.pressure[indices])
        
        # Calculate velocity divergence on the grid
        vel_grid_x = np.zeros((GRID_SIZE, GRID_SIZE))
        vel_grid_y = np.zeros((GRID_SIZE, GRID_SIZE))
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                indices = np.where((np.floor(self.pos[:,0]) == i) & (np.floor(self.pos[:,1]) == j))[0]
                if len(indices) > 0:
                    vel_grid_x[j, i] = np.mean(self.vel[indices, 0])
                    vel_grid_y[j, i] = np.mean(self.vel[indices, 1])
        
        grad_x = np.gradient(vel_grid_x, axis=1)
        grad_y = np.gradient(vel_grid_y, axis=0)
        vel_div_grid = grad_x + grad_y

        return density_grid, pressure_grid, vel_div_grid


# --- Stable Fluids Implementation ---

class StableFluidsSolver:
    """A simple 2D grid-based Stable Fluids solver."""
    def __init__(self):
        self.size = GRID_SIZE
        self.u = np.zeros((self.size, self.size))
        self.v = np.zeros((self.size, self.size))
        self.density = np.zeros((self.size, self.size))
        
        # Initialize dam break scenario
        self.density[int(self.size*0.2):int(self.size*0.8), :int(self.size*0.4)] = 1.0

    def _advect(self, d, u, v):
        rows, cols = np.indices((self.size, self.size))
        # Backtrace
        x = np.clip(cols - u * DT * self.size, 0, self.size - 1.01)
        y = np.clip(rows - v * DT * self.size, 0, self.size - 1.01)
        
        # Bilinear interpolation
        i0, i1 = np.floor(x).astype(int), np.ceil(x).astype(int)
        j0, j1 = np.floor(y).astype(int), np.ceil(y).astype(int)
        s1, s0 = x - i0, 1 - (x - i0)
        t1, t0 = y - j0, 1 - (y - j0)
        
        return (s0 * (t0 * d[j0, i0] + t1 * d[j1, i0]) +
                s1 * (t0 * d[j0, i1] + t1 * d[j1, i1]))

    def _project(self):
        div = -0.5 * (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1) +
                      np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0)) / self.size
        p = np.zeros((self.size, self.size)) # Pressure
        
        # Jacobi iteration to solve Poisson equation
        for _ in range(20):
            p_old = p.copy()
            p = (np.roll(p_old, 1, axis=1) + np.roll(p_old, -1, axis=1) +
                 np.roll(p_old, 1, axis=0) + np.roll(p_old, -1, axis=0) - div * self.size**2) / 4.0
        
        self.u -= 0.5 * self.size * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
        self.v -= 0.5 * self.size * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))
        return p

    def step(self):
        """Perform one step of the Stable Fluids simulation."""
        self.v += GRAVITY[1] * DT  # Add gravity
        
        u_prev, v_prev = self.u.copy(), self.v.copy()
        self.u = self._advect(u_prev, u_prev, v_prev)
        self.v = self._advect(v_prev, u_prev, v_prev)
        
        pressure = self._project()
        
        self.density = self._advect(self.density, self.u, self.v)
        return pressure
        
    def get_gridded_data(self, pressure):
        """Return the gridded data fields."""
        # Velocity divergence
        grad_u = np.gradient(self.u, axis=1)
        grad_v = np.gradient(self.v, axis=0)
        vel_div = grad_u + grad_v
        
        return self.density, pressure, vel_div


# --- Visualization ---

def create_comparison_plot(sph_data, sf_data):
    """Creates a multi-panel plot comparing SPH and Stable Fluids."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle('Static Analysis of 2D Water Simulation Models (Dam Break)', fontsize=22)

    sph_density, sph_pressure, sph_vel_div = sph_data
    sf_density, sf_pressure, sf_vel_div = sf_data

    titles = ['Density ($\\rho$)', 'Pressure ($P$)', 'Velocity Divergence ($\\nabla \\cdot \\mathbf{v}$)']
    data_pairs = [
        ((sph_density, sf_density), SEQUENTIAL_CMAP),
        ((sph_pressure, sf_pressure), DIVERGING_CMAP),
        ((sph_vel_div, sf_vel_div), DIVERGING_CMAP)
    ]

    for i, ((sph_d, sf_d), cmap) in enumerate(data_pairs):
        # SPH Plot
        ax_sph = axes[0, i]
        im_sph = ax_sph.imshow(sph_d, origin='lower', cmap=cmap, interpolation='bilinear')
        ax_sph.set_title(f'SPH: {titles[i]}', fontsize=16)
        fig.colorbar(im_sph, ax=ax_sph, orientation='vertical', fraction=0.046, pad=0.04)

        # Stable Fluids Plot
        ax_sf = axes[1, i]
        im_sf = ax_sf.imshow(sf_d, origin='lower', cmap=cmap, interpolation='bilinear')
        ax_sf.set_title(f'Stable Fluids: {titles[i]}', fontsize=16)
        fig.colorbar(im_sf, ax=ax_sf, orientation='vertical', fraction=0.046, pad=0.04)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    print(f"Saving comparison plot to {OUTPUT_FILENAME}")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
    plt.savefig(OUTPUT_FILENAME, dpi=200, bbox_inches='tight')
    plt.close(fig)


# --- Main Execution ---

if __name__ == "__main__":
    print("--- Starting Static Visual Analysis ---")

    # 1. Run SPH Simulation
    print("Running SPH simulation...")
    sph_solver = SPHSolver()
    for step in range(SIMULATION_STEPS):
        sph_solver.step()
        print(f"  SPH Step {step+1}/{SIMULATION_STEPS}", end='\\r')
    print("\\nSPH simulation finished.")
    sph_gridded_data = sph_solver.get_gridded_data()
    print("Generated gridded data from SPH particles.")

    # 2. Run Stable Fluids Simulation
    print("Running Stable Fluids simulation...")
    sf_solver = StableFluidsSolver()
    final_pressure = None
    for step in range(SIMULATION_STEPS):
        final_pressure = sf_solver.step()
        print(f"  Stable Fluids Step {step+1}/{SIMULATION_STEPS}", end='\\r')
    print("\\nStable Fluids simulation finished.")
    sf_gridded_data = sf_solver.get_gridded_data(final_pressure)
    print("Generated gridded data from Stable Fluids solver.")

    # 3. Create and save the comparison plot
    print("Creating comparison plot...")
    create_comparison_plot(sph_gridded_data, sf_gridded_data)

    print(f"--- Static Visual Analysis Complete. Output saved to {OUTPUT_FILENAME} ---") 
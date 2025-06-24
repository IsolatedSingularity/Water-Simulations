"""
Advanced Fluid Dynamics Simulations

This script generates three distinct, advanced fluid dynamics simulations based on
concepts from fluid dynamics literature. Each simulation is saved as a separate
GIF animation, highlighting a unique aspect of fluid flow with a specific
color palette.

1.  **Vortex Shedding (Kármán Vortex Street):** Demonstrates the creation of
    alternating vortices as a fluid flows past a solid obstacle.
2.  **Viscosity Comparison:** Shows a side-by-side comparison of two fluids
    with different viscosities in a "dam break" scenario using an SPH solver.
3.  **Wave Tank:** Simulates the propagation of surface waves in a tank,
    driven by a sinusoidal oscillator, using a height-field method.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# --- General Parameters ---
PLOT_DIR = 'Plots'
if not os.path.exists(PLOT_DIR):
    os.makedirs(PLOT_DIR)


# --- Simulation 1: Vortex Shedding ---

class VortexSolver:
    """A grid-based solver for simulating vortex shedding."""
    def __init__(self, size=(256, 64)):
        self.width, self.height = size
        self.u = np.zeros((self.height, self.width))
        self.v = np.zeros((self.height, self.width))
        self.density = np.zeros((self.height, self.width))
        self.pressure = np.zeros((self.height, self.width))

        # Obstacle properties
        c_x, c_y = self.width // 5, self.height // 2
        rad = self.height // 10
        y, x = np.ogrid[-c_y:self.height - c_y, -c_x:self.width - c_x]
        self.obstacle = x*x + y*y <= rad*rad
        
        # Inflow velocity
        self.u[:, 0] = 1.0

    def _advect(self, d, u, v):
        rows, cols = np.indices((self.height, self.width))
        x = np.clip(cols - u * 1.0, 0, self.width - 1.01)
        y = np.clip(rows - v * 1.0, 0, self.height - 1.01)
        i0, i1 = np.floor(x).astype(int), np.ceil(x).astype(int)
        j0, j1 = np.floor(y).astype(int), np.ceil(y).astype(int)
        s1, s0 = x - i0, 1 - (x - i0)
        t1, t0 = y - j0, 1 - (y - j0)
        return (s0 * (t0 * d[j0, i0] + t1 * d[j1, i0]) +
                s1 * (t0 * d[j0, i1] + t1 * d[j1, i1]))

    def _diffuse(self, d, viscosity, dt=1.0):
        """Apply diffusion to a field using Jacobi iteration."""
        d_new = d.copy()
        for _ in range(5):  # Fewer iterations for a small amount of diffusion
            d_new = (d + viscosity * (np.roll(d, 1, axis=1) + np.roll(d, -1, axis=1) +
                                    np.roll(d, 1, axis=0) + np.roll(d, -1, axis=0))) / (1 + 4 * viscosity)
            d_new[self.obstacle] = 0  # Maintain obstacle boundary
        return d_new

    def _project(self):
        div = -0.5 * (np.roll(self.u, -1, axis=1) - np.roll(self.u, 1, axis=1) +
                      np.roll(self.v, -1, axis=0) - np.roll(self.v, 1, axis=0))
        p = self.pressure
        for _ in range(30):
            p_old = p.copy()
            p = (np.roll(p_old, 1, axis=1) + np.roll(p_old, -1, axis=1) +
                 np.roll(p_old, 1, axis=0) + np.roll(p_old, -1, axis=0) - div) / 4.0
            p[self.obstacle] = 0
        self.pressure = p
        self.u -= 0.5 * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
        self.v -= 0.5 * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))

    def step(self):
        # Advect velocity
        self.u = self._advect(self.u, self.u, self.v)
        self.v = self._advect(self.v, self.u, self.v)
        
        # Add diffusion for stability
        self.u = self._diffuse(self.u, 0.01)
        self.v = self._diffuse(self.v, 0.01)
        
        self._project()
        
        # Enforce boundary and obstacle conditions
        self.u[:, 0] = 1.0; self.v[:, 0] = 0
        self.u[self.obstacle] = 0; self.v[self.obstacle] = 0

    def get_vorticity(self):
        return (np.roll(self.v, -1, axis=1) - np.roll(self.v, 1, axis=1) -
                np.roll(self.u, -1, axis=0) + np.roll(self.u, 1, axis=0))

def generate_vortex_simulation():
    print("--- 1. Generating Kármán Vortex Street Simulation ---")
    FRAMES = 200
    FILENAME = os.path.join(PLOT_DIR, 'vortex_street.gif')
    solver = VortexSolver()
    fig, ax = plt.subplots(figsize=(10, 2.5))
    vorticity_cmap = sns.diverging_palette(240, 10, s=80, l=55, n=256, as_cmap=True)
    
    im = ax.imshow(solver.get_vorticity(), cmap=vorticity_cmap, vmin=-0.1, vmax=0.1)
    obstacle_circle = plt.Circle((solver.width // 5, solver.height // 2), solver.height // 10, color='black')
    ax.add_artist(obstacle_circle)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Vortex Shedding (Kármán Vortex Street)")

    def update(frame):
        # Slower evolution per frame for better stability
        solver.step()
        vorticity = np.nan_to_num(solver.get_vorticity(), nan=0.0, posinf=0.1, neginf=-0.1)
        im.set_data(vorticity)
        print(f"  Generating frame {frame+1}/{FRAMES}", end='\\r')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, blit=True)
    ani.save(FILENAME, writer='pillow', fps=30)
    plt.close(fig)
    print(f"\n--- Vortex simulation saved to {FILENAME} ---")


# --- Simulation 2: Viscosity Comparison ---

class SPHSolverForViscosity:
    """An SPH solver adapted for viscosity demonstrations."""
    def __init__(self, viscosity, particle_count=400):
        self.viscosity = viscosity
        self.n_particles = particle_count
        self.pos = (np.random.rand(self.n_particles, 2) * 0.4 + 0.1) * np.array([1, 2])
        self.vel = np.zeros_like(self.pos)
        self.rho = np.zeros(self.n_particles)
        self.pressure = np.zeros(self.n_particles)
        self.mass = np.ones(self.n_particles) * 1.0
        self.h = 0.1 # Smoothing radius
        self.g = np.array([0, -9.8])
        self.dt = 0.005

    def step(self):
        # Simplified SPH update (without full pressure/density solve for speed)
        # Compute viscosity forces
        f_visc = np.zeros_like(self.pos)
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i == j: continue
                r_vec = self.pos[i] - self.pos[j]
                r = np.linalg.norm(r_vec)
                if r < self.h:
                    # Simplified viscosity force (laplacian of velocity)
                    f_visc[i] += self.viscosity * (self.vel[j] - self.vel[i]) * (self.h - r)

        self.vel += (self.g + f_visc) * self.dt
        self.pos += self.vel * self.dt

        # Damping to prevent runaway velocities
        self.vel *= 0.99

        # Boundary conditions
        self.pos = np.clip(self.pos, 0.01, 1.99)
        self.vel[self.pos[:, 0] <= 0.01, 0] *= -0.5
        self.vel[self.pos[:, 0] >= 1.99, 0] *= -0.5
        self.vel[self.pos[:, 1] <= 0.01, 1] *= -0.5
        self.vel[self.pos[:, 1] >= 1.99, 1] *= -0.5

def generate_viscosity_comparison():
    print("--- 2. Generating Viscosity Comparison Simulation ---")
    FRAMES = 150
    FILENAME = os.path.join(PLOT_DIR, 'viscosity_comparison.gif')
    
    # Two solvers with different viscosities
    solver_low_visc = SPHSolverForViscosity(viscosity=1.0)
    solver_high_visc = SPHSolverForViscosity(viscosity=80.0)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    visc_cmap = sns.color_palette("crest", as_cmap=True)

    scat_low = axes[0].scatter([], [], s=15, cmap=visc_cmap)
    scat_high = axes[1].scatter([], [], s=15, cmap=visc_cmap)

    for ax, title in zip(axes, ["Low Viscosity", "High Viscosity"]):
        ax.set_xlim(0, 2); ax.set_ylim(0, 2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', 'box')
        ax.set_title(title)
    fig.suptitle("SPH Simulation: Viscosity Comparison")

    def update(frame):
        solver_low_visc.step()
        solver_high_visc.step()
        
        for scat, solver in [(scat_low, solver_low_visc), (scat_high, solver_high_visc)]:
            vel_mag = np.linalg.norm(solver.vel, axis=1)
            scat.set_offsets(solver.pos)
            scat.set_array(vel_mag)
            # Use dynamic color limits
            if vel_mag.max() > 1e-6:
                scat.set_clim(0, vel_mag.max())
            else:
                scat.set_clim(0, 1)
        
        print(f"  Generating frame {frame+1}/{FRAMES}", end='\\r')
        return scat_low, scat_high

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, blit=True)
    ani.save(FILENAME, writer='pillow', fps=30)
    plt.close(fig)
    print(f"\n--- Viscosity simulation saved to {FILENAME} ---")


# --- Simulation 3: Wave Tank ---

class WaveSolver:
    """A height-field based solver for wave propagation."""
    def __init__(self, size=(128, 128)):
        self.width, self.height = size
        self.h = np.zeros(size) # Height field
        self.v = np.zeros(size) # Vertical velocity
        self.dt = 0.1

    def step(self, frame):
        # Calculate laplacian of height field
        laplacian_h = (np.roll(self.h, 1, axis=0) + np.roll(self.h, -1, axis=0) +
                       np.roll(self.h, 1, axis=1) + np.roll(self.h, -1, axis=1) - 4 * self.h)
        
        self.v += laplacian_h * 0.2 # Increased acceleration for faster ripples
        self.v *= 0.98 # Slightly more damping to control the faster waves
        self.h += self.v * self.dt

        # Multiple sinusoidal drivers
        driver_freq = 0.3
        self.h[self.height // 4, 5] = np.sin(frame * driver_freq + np.pi) * 1.5
        self.h[self.height // 2, 5] = np.sin(frame * driver_freq) * 2.0
        self.h[3 * self.height // 4, 5] = np.sin(frame * driver_freq + np.pi / 2) * 1.5

    def get_height_field(self):
        return self.h

def generate_wave_simulation():
    print("--- 3. Generating Wave Tank Simulation ---")
    FRAMES = 600 # Doubled frame count for a longer animation
    FILENAME = os.path.join(PLOT_DIR, 'wave_tank.gif')
    solver = WaveSolver()
    fig, ax = plt.subplots(figsize=(8, 8))
    wave_cmap = sns.color_palette("YlGnBu", as_cmap=True)
    
    im = ax.imshow(solver.get_height_field(), cmap=wave_cmap, vmin=-1, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Height-Field Wave Tank")

    def update(frame):
        solver.step(frame)
        im.set_data(solver.get_height_field())
        print(f"  Generating frame {frame+1}/{FRAMES}", end='\\r')
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=FRAMES, blit=True)
    ani.save(FILENAME, writer='pillow', fps=30)
    plt.close(fig)
    print(f"\n--- Wave simulation saved to {FILENAME} ---")

if __name__ == "__main__":
    generate_vortex_simulation()
    generate_viscosity_comparison()
    generate_wave_simulation()
    print("\nAll advanced simulations generated successfully.")
    print("\nPhase v) complete. Awaiting confirmation for Phase vi).") 
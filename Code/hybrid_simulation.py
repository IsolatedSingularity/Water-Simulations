"""
Hybrid PIC/FLIP Fluid Simulation

This script implements a hybrid particle-in-cell (PIC) fluid simulation, specifically
using a FLIP (Fluid-Implicit-Particle) method. It combines the advantages of
particle-based methods (Lagrangian, no numerical diffusion for advection) and
grid-based methods (Eulerian, efficient pressure solves).

- Particles carry velocity and are used for advection.
- A grid is used to compute pressure forces and enforce incompressibility.

The script generates an animation of a "dam break" scenario and saves it as a GIF.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# --- Simulation Parameters ---
GRID_SIZE = 128  # Increased resolution for more detail
PARTICLES_PER_ROW = 50  # Denser particle packing
DT = 0.04  # Slightly smaller timestep for CFL stability
GRAVITY = np.array([0, -9.8 * 2])  # Stronger gravity for a more dynamic splash
ANIMATION_FRAMES = 300  # Longer simulation to see full evolution
FLIP_ALPHA = 0.95  # Blending factor: 1.0=Pure FLIP, 0.0=Pure PIC

# --- Visualization Parameters ---
PLOT_DIR = 'Plots'
OUTPUT_FILENAME = os.path.join(PLOT_DIR, 'hybrid_simulation.gif')
SEQUENTIAL_CMAP = sns.color_palette("mako", as_cmap=True)

class HybridSolver:
    """A 2D PIC/FLIP-based hybrid fluid solver."""
    def __init__(self):
        # Particle state
        x = np.linspace(0.1 * GRID_SIZE, GRID_SIZE * 0.4, PARTICLES_PER_ROW)
        y = np.linspace(0.1 * GRID_SIZE, GRID_SIZE * 0.8, PARTICLES_PER_ROW * 2)
        xx, yy = np.meshgrid(x, y)
        self.pos = np.vstack([xx.ravel(), yy.ravel()]).T.astype(float)
        self.vel = np.zeros_like(self.pos)
        self.n_particles = len(self.pos)
        
        # Grid state
        self.grid_u = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1))
        self.grid_v = np.zeros((GRID_SIZE + 1, GRID_SIZE + 1))

    def _particles_to_grid(self):
        """Transfer particle velocities to the grid (P2G)."""
        self.grid_u.fill(0)
        self.grid_v.fill(0)
        grid_mass_u = np.zeros_like(self.grid_u)
        grid_mass_v = np.zeros_like(self.grid_v)

        # Bilinear interpolation weights
        x, y = self.pos[:, 0], self.pos[:, 1]
        i, j = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - i, y - j

        w00 = (1 - fx) * (1 - fy)
        w10 = fx * (1 - fy)
        w01 = (1 - fx) * fy
        w11 = fx * fy

        for p_idx in range(self.n_particles):
            ip, jp = i[p_idx], j[p_idx]
            up, vp = self.vel[p_idx]

            # Add weighted velocity and mass to surrounding 4 grid nodes
            grid_mass_u[jp, ip] += w00[p_idx]; self.grid_u[jp, ip] += w00[p_idx] * up
            grid_mass_v[jp, ip] += w00[p_idx]; self.grid_v[jp, ip] += w00[p_idx] * vp
            
            grid_mass_u[jp, ip + 1] += w10[p_idx]; self.grid_u[jp, ip + 1] += w10[p_idx] * up
            grid_mass_v[jp, ip + 1] += w10[p_idx]; self.grid_v[jp, ip + 1] += w10[p_idx] * vp

            grid_mass_u[jp + 1, ip] += w01[p_idx]; self.grid_u[jp + 1, ip] += w01[p_idx] * up
            grid_mass_v[jp + 1, ip] += w01[p_idx]; self.grid_v[jp + 1, ip] += w01[p_idx] * vp

            grid_mass_u[jp + 1, ip + 1] += w11[p_idx]; self.grid_u[jp + 1, ip + 1] += w11[p_idx] * up
            grid_mass_v[jp + 1, ip + 1] += w11[p_idx]; self.grid_v[jp + 1, ip + 1] += w11[p_idx] * vp
        
        # Normalize velocities
        self.grid_u[grid_mass_u > 0] /= grid_mass_u[grid_mass_u > 0]
        self.grid_v[grid_mass_v > 0] /= grid_mass_v[grid_mass_v > 0]

    def _project(self):
        """Project the grid velocity field to be divergence-free."""
        div = -0.5 * (np.roll(self.grid_u, -1, axis=1) - np.roll(self.grid_u, 1, axis=1) +
                      np.roll(self.grid_v, -1, axis=0) - np.roll(self.grid_v, 1, axis=0))
        p = np.zeros_like(self.grid_u) # Pressure
        
        for _ in range(40): # More iterations for better convergence
            p_old = p.copy()
            p = (np.roll(p_old, 1, axis=1) + np.roll(p_old, -1, axis=1) +
                 np.roll(p_old, 1, axis=0) + np.roll(p_old, -1, axis=0) - div) / 4.0
        
        self.grid_u -= 0.5 * (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1))
        self.grid_v -= 0.5 * (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0))

    def _grid_to_particles(self, old_grid_u, old_grid_v):
        """Update particle velocities from grid using a FLIP/PIC blend."""
        x, y = self.pos[:, 0], self.pos[:, 1]
        i, j = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = x - i, y - j

        # Interpolate grid velocities (for PIC part)
        pic_vel_u = (1-fx)*(1-fy)*self.grid_u[j,i] + fx*(1-fy)*self.grid_u[j,i+1] + (1-fx)*fy*self.grid_u[j+1,i] + fx*fy*self.grid_u[j+1,i+1]
        pic_vel_v = (1-fx)*(1-fy)*self.grid_v[j,i] + fx*(1-fy)*self.grid_v[j,i+1] + (1-fx)*fy*self.grid_v[j+1,i] + fx*fy*self.grid_v[j+1,i+1]
        
        # Interpolate change in grid velocities (for FLIP part)
        du = self.grid_u - old_grid_u
        dv = self.grid_v - old_grid_v
        du_interp = (1-fx)*(1-fy)*du[j,i] + fx*(1-fy)*du[j,i+1] + (1-fx)*fy*du[j+1,i] + fx*fy*du[j+1,i+1]
        dv_interp = (1-fx)*(1-fy)*dv[j,i] + fx*(1-fy)*dv[j,i+1] + (1-fx)*fy*dv[j+1,i] + fx*fy*dv[j+1,i+1]
        
        # Current particle velocity + change is the FLIP velocity
        flip_vel_u = self.vel[:, 0] + du_interp
        flip_vel_v = self.vel[:, 1] + dv_interp

        # Final velocity is a blend of FLIP and PIC
        self.vel[:, 0] = FLIP_ALPHA * flip_vel_u + (1.0 - FLIP_ALPHA) * pic_vel_u
        self.vel[:, 1] = FLIP_ALPHA * flip_vel_v + (1.0 - FLIP_ALPHA) * pic_vel_v

    def _apply_boundary_conditions(self):
        """Applies solid boundary conditions with elastic collision response."""
        boundary_padding = 1.0
        damping = -0.5  # Controls bounciness

        # Create boolean masks for particles outside boundaries
        out_left = self.pos[:, 0] < boundary_padding
        out_right = self.pos[:, 0] > GRID_SIZE - boundary_padding
        out_bottom = self.pos[:, 1] < boundary_padding
        out_top = self.pos[:, 1] > GRID_SIZE - boundary_padding

        # Update positions and velocities using masks
        self.pos[out_left, 0] = boundary_padding
        self.vel[out_left, 0] *= damping
        
        self.pos[out_right, 0] = GRID_SIZE - boundary_padding
        self.vel[out_right, 0] *= damping

        self.pos[out_bottom, 1] = boundary_padding
        self.vel[out_bottom, 1] *= damping
        
        self.pos[out_top, 1] = GRID_SIZE - boundary_padding
        self.vel[out_top, 1] *= damping

    def step(self):
        """Perform one full step of the PIC/FLIP simulation."""
        self._particles_to_grid()
        
        old_grid_u = self.grid_u.copy()
        old_grid_v = self.grid_v.copy()
        
        # Grid Operations
        self.grid_v += GRAVITY[1] * DT
        self._project()
        
        self._grid_to_particles(old_grid_u, old_grid_v)
        
        # Advect particles
        self.pos += self.vel * DT
        
        # Enforce boundary collisions
        self._apply_boundary_conditions()

    def get_particle_positions(self):
        return self.pos

if __name__ == "__main__":
    print("--- Starting Hybrid PIC/FLIP Fluid Simulation ---")
    
    solver = HybridSolver()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor(SEQUENTIAL_CMAP(0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title("Hybrid PIC/FLIP Fluid Simulation")

    # Use scatter plot for particles
    scatter = ax.scatter([], [], s=10, c=[], cmap=SEQUENTIAL_CMAP, vmin=0)

    def update(frame):
        solver.step()
        positions = solver.get_particle_positions()
        velocities = np.linalg.norm(solver.vel, axis=1)
        
        scatter.set_offsets(positions)
        scatter.set_array(velocities)
        
        # Dynamically adjust color map limits for better visualization
        max_vel = np.max(velocities)
        if max_vel > 1e-6:
            scatter.set_clim(0, max_vel)
        else:
            scatter.set_clim(0, 1.0)
        
        print(f"  Generating frame {frame+1}/{ANIMATION_FRAMES}, Particles: {solver.n_particles} ", end='\\r')
        return [scatter]

    print("Initializing animation...")
    ani = animation.FuncAnimation(fig, update, frames=ANIMATION_FRAMES, blit=True)

    print(f"\\nSaving animation to {OUTPUT_FILENAME}...")
    if not os.path.exists(PLOT_DIR):
        os.makedirs(PLOT_DIR)
        
    ani.save(OUTPUT_FILENAME, writer='pillow', fps=30, dpi=100)
    plt.close(fig)
    
    print("--- Hybrid simulation animation saved successfully! ---")

    print("\\nPhase v) complete. Awaiting confirmation for Phase vi).") 